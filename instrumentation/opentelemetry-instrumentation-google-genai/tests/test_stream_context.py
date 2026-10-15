# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import nullcontext
from functools import wraps

import pytest
from google.genai import Client
from google.genai.models import AsyncModels, Models
from google.genai.types import GenerateContentResponse

from opentelemetry import context, trace
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
    AsyncInteractionsResource,
    InteractionsResource,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import Span, StatusCode

REQUIRES_INTERACTIONS = pytest.mark.skipif(
    not _HAS_INTERACTIONS,
    reason="Interactions are not supported in this version of google-genai",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expected_operation",
    [
        pytest.param("generate_content", id="models.generate_content_stream"),
        pytest.param(
            "interactions.create",
            id="interactions.create-model",
            marks=REQUIRES_INTERACTIONS,
        ),
        pytest.param(
            "invoke_agent",
            id="interactions.create-agent",
            marks=REQUIRES_INTERACTIONS,
        ),
        pytest.param(
            "fetch_response",
            id="interactions.get",
            marks=REQUIRES_INTERACTIONS,
        ),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ending", ["drain", "caller_error", "stream_error"])
async def test_stream_context(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    expected_operation: str,
    asynchronous: bool,
    ending: str,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    sdk_span: Span | None = None
    error = ConnectionError("stream failed")
    chunk = (
        GenerateContentResponse(response_id="response-id", candidates=[])
        if expected_operation == "generate_content"
        else {"event_type": "content"}
    )

    def produce() -> Iterator[object]:
        for _ in range(2):
            assert trace.get_current_span() is sdk_span
            yield chunk
            if ending == "stream_error":
                raise error

    async def aproduce() -> AsyncIterator[object]:
        for item in produce():
            yield item

    resource_type = (
        (AsyncModels if asynchronous else Models)
        if expected_operation == "generate_content"
        else (
            AsyncInteractionsResource if asynchronous else InteractionsResource
        )
    )
    method_name = (
        "generate_content_stream"
        if expected_operation == "generate_content"
        else "get"
        if expected_operation == "fetch_response"
        else "create"
    )
    original = getattr(resource_type, method_name)
    source = aproduce() if asynchronous else produce()

    @wraps(original)
    def call(
        *args: object, **kwargs: object
    ) -> Iterator[object] | AsyncIterator[object]:
        nonlocal sdk_span
        sdk_span = trace.get_current_span()
        return source

    @wraps(original)
    async def acall(
        *args: object, **kwargs: object
    ) -> Iterator[object] | AsyncIterator[object]:
        nonlocal sdk_span
        sdk_span = trace.get_current_span()
        return source

    monkeypatch.setattr(
        resource_type, method_name, acall if asynchronous else call
    )
    # The instrumentor sets this independently of the shared content setting.
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", "true")
    with (
        instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture="NO_CONTENT",
        ),
        tracer.start_as_current_span("caller") as caller,
    ):
        client = Client(api_key="test-key")
        before = context.get_current()
        api = client.aio if asynchronous else client
        if expected_operation == "generate_content":
            stream = api.models.generate_content_stream(
                model="gemini-2.5-flash", contents="Hello"
            )
        elif expected_operation == "fetch_response":
            stream = api.interactions.get("response-id", stream=True)
        elif expected_operation == "invoke_agent":
            stream = api.interactions.create(
                agent="test-agent", input="Hello", stream=True
            )
        else:
            stream = api.interactions.create(
                model="gemini-2.5-flash", input="Hello", stream=True
            )
        if asynchronous:
            stream = await stream
        assert sdk_span is not None
        assert sdk_span.is_recording()
        assert context.get_current() is before
        received: list[object] = []

        def check_caller_context(item: object) -> None:
            assert item is chunk
            assert context.get_current() is before
            assert sdk_span.is_recording()
            received.append(item)
            if ending == "caller_error":
                raise error

        with (
            pytest.raises(ConnectionError)
            if ending != "drain"
            else nullcontext()
        ) as raised:
            if asynchronous:

                async def consume() -> None:
                    try:
                        async with stream:
                            async for item in stream:
                                check_caller_context(item)
                    finally:
                        assert context.get_current() is before

                await asyncio.create_task(consume())
            else:
                with stream:
                    for item in stream:
                        check_caller_context(item)

        if raised is not None:
            assert raised.value is error
        assert len(received) == (2 if ending == "drain" else 1)
        assert context.get_current() is before
        assert not sdk_span.is_recording()

    invocation_span, caller_span = span_exporter.get_finished_spans()
    assert invocation_span.context == sdk_span.get_span_context()
    assert invocation_span.parent == caller.get_span_context()
    assert caller_span.context == caller.get_span_context()
    assert invocation_span.attributes is not None
    assert (
        invocation_span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == expected_operation
    )
    failed = ending != "drain"
    assert invocation_span.status.status_code == (
        StatusCode.ERROR if failed else StatusCode.UNSET
    )
    if failed:
        assert invocation_span.attributes[ERROR_TYPE] == "ConnectionError"
    else:
        assert ERROR_TYPE not in invocation_span.attributes
