# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator, Iterator
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
from opentelemetry.semconv.attributes.error_attributes import (
    ERROR_TYPE,
    ErrorTypeValues,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import Span, StatusCode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "generate_content",
        "interactions.create",
        "invoke_agent",
        "fetch_response",
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "ending",
    [
        "drain",
        "empty",
        "close",
        "caller_error",
        "stream_error",
        "close_error",
        "cancel",
        "abandon",
        "abandon_unread",
    ],
)
async def test_stream_context(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    operation: str,
    asynchronous: bool,
    ending: str,
) -> None:
    if operation != "generate_content" and not _HAS_INTERACTIONS:
        pytest.skip("This SDK has no Interactions API")

    tracer = tracer_provider.get_tracer(__name__)
    sdk_span: Span | None = None
    sdk_steps: list[str] = []
    error = (
        asyncio.CancelledError("stream cancelled")
        if ending == "cancel"
        else ConnectionError("stream failed")
    )
    chunk = (
        GenerateContentResponse(response_id="response-id", candidates=[])
        if operation == "generate_content"
        else {"event_type": "content"}
    )

    def check_sdk_context(step: str) -> None:
        assert trace.get_current_span() is sdk_span
        sdk_steps.append(step)
        with tracer.start_as_current_span(f"sdk.{step}"):
            pass

    def produce() -> Iterator[object]:
        try:
            check_sdk_context("read")
            if ending == "empty":
                return
            yield chunk
            check_sdk_context("read")
            if ending in ("stream_error", "cancel"):
                raise error
            yield chunk
        finally:
            if ending != "abandon":
                check_sdk_context("close")
                if ending == "close_error":
                    raise error

    async def aproduce() -> AsyncIterator[object]:
        source = produce()
        try:
            for item in source:
                await asyncio.sleep(0)
                yield item
        finally:
            source.close()

    resource_type = (
        (AsyncModels if asynchronous else Models)
        if operation == "generate_content"
        else (
            AsyncInteractionsResource if asynchronous else InteractionsResource
        )
    )
    method_name = (
        "generate_content_stream"
        if operation == "generate_content"
        else "get"
        if operation == "fetch_response"
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
        if operation == "generate_content":
            stream = api.models.generate_content_stream(
                model="gemini-2.5-flash", contents="Hello"
            )
        elif operation == "fetch_response":
            stream = api.interactions.get("response-id", stream=True)
        elif operation == "invoke_agent":
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
        assert sdk_steps == []
        assert isinstance(stream, type(source))
        assert stream.__class__ is source.__class__

        if ending not in ("empty", "abandon_unread"):
            if asynchronous:
                assert await asyncio.create_task(anext(stream)) is chunk
            else:
                assert next(stream) is chunk
            assert context.get_current() is before
            assert sdk_span.is_recording()

        with tracer.start_as_current_span("caller.work"):
            pass

        if ending.startswith("abandon"):
            del stream
            gc.collect()
        elif ending == "caller_error":
            with pytest.raises(ConnectionError) as raised:
                if asynchronous:

                    async def fail_in_caller() -> None:
                        async with stream:
                            assert context.get_current() is before
                            raise error

                    await asyncio.create_task(fail_in_caller())
                else:
                    with stream:
                        assert context.get_current() is before
                        raise error
            assert raised.value is error
        elif ending in ("stream_error", "close_error", "cancel"):
            if asynchronous:

                async def fail_in_sdk() -> None:
                    with pytest.raises(type(error)) as raised:
                        if ending == "close_error":
                            await stream.aclose()
                        else:
                            await anext(stream)
                    # Python 3.10 recreates CancelledError across task boundaries.
                    assert raised.value is error
                    assert context.get_current() is before

                await asyncio.create_task(fail_in_sdk())
            else:
                with pytest.raises(type(error)) as raised:
                    if ending == "close_error":
                        stream.close()
                    else:
                        next(stream)
                assert raised.value is error
        elif ending == "close":
            if asynchronous:
                await asyncio.create_task(stream.aclose())
            else:
                stream.close()
        else:
            remaining = [] if ending == "empty" else [chunk]
            if asynchronous:

                async def drain() -> list[object]:
                    return [item async for item in stream]

                assert await asyncio.create_task(drain()) == remaining
            else:
                assert list(stream) == remaining

        assert context.get_current() is before
        assert not sdk_span.is_recording()
        if not ending.startswith("abandon"):
            assert sdk_steps[-1] == "close"
            if asynchronous:
                await asyncio.create_task(stream.aclose())
            else:
                stream.close()
        else:
            if asynchronous:
                await source.aclose()
            else:
                source.close()
        assert context.get_current() is before

    spans = span_exporter.get_finished_spans()
    invocation_spans = [
        span for span in spans if span.context == sdk_span.get_span_context()
    ]
    assert len(invocation_spans) == 1
    (invocation_span,) = invocation_spans
    assert invocation_span.parent == caller.get_span_context()
    assert invocation_span.attributes is not None
    assert (
        invocation_span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == operation
    )
    failed = ending not in ("drain", "empty", "close")
    assert invocation_span.status.status_code == (
        StatusCode.ERROR if failed else StatusCode.UNSET
    )
    if failed:
        expected_error = (
            ErrorTypeValues.OTHER.value
            if ending.startswith("abandon")
            else "asyncio.exceptions.CancelledError"
            if ending == "cancel"
            else type(error).__name__
        )
        assert invocation_span.attributes[ERROR_TYPE] == expected_error
    else:
        assert ERROR_TYPE not in invocation_span.attributes
    for span in spans:
        if span.name.startswith("sdk."):
            assert span.parent == invocation_span.context
        elif span.name == "caller.work":
            assert span.parent == caller.get_span_context()
    assert not [
        record
        for record in caplog.records
        if record.name == "opentelemetry.context" and record.levelno >= 40
    ]
