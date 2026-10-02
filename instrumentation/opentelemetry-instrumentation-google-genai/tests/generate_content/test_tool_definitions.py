# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import httpx
import pytest
from google.genai import Client, types

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.generate_content import (
    _tool_to_tool_definition,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.test_util_genai.instrumentor import instrument

_JSON_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string", "enum": ["Paris", "London"]}},
    "required": ["city"],
    "additionalProperties": False,
}
_PARAMETERS = {
    "type": "OBJECT",
    "properties": {"city": {"type": "STRING"}},
    "required": ["city"],
}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("config_dict", [False, True])
@pytest.mark.parametrize("capture_content", [False, True])
@pytest.mark.parametrize(
    "parameter_fields,expected_parameters",
    [
        pytest.param(
            {"parameters": _PARAMETERS}, _PARAMETERS, id="parameters"
        ),
        pytest.param(
            {"parameters_json_schema": _JSON_SCHEMA},
            _JSON_SCHEMA,
            id="json-schema",
        ),
        pytest.param(
            {"parameters_json_schema": {}}, {}, id="empty-json-schema"
        ),
        pytest.param({}, None, id="no-parameters"),
    ],
)
@pytest.mark.asyncio
async def test_tool_definition_parameters(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    streaming: bool,
    config_dict: bool,
    capture_content: bool,
    parameter_fields: dict[str, object],
    expected_parameters: dict[str, object] | None,
) -> None:
    declaration = types.FunctionDeclaration(
        name="get_weather",
        description="Get the weather for a city.",
        **parameter_fields,
    )
    original_declaration = declaration.model_dump(mode="json")
    config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[declaration])]
    )
    requests: list[httpx.Request] = []
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "Sunny."}]},
                "finishReason": "STOP",
            }
        ],
        "modelVersion": "gemini-2.5-flash",
    }

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if streaming:
            return httpx.Response(
                200,
                text=f"data: {json.dumps(response)}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(respond)
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY" if capture_content else "NO_CONTENT",
    ):
        with Client(
            api_key="test-key",
            vertexai=False,
            http_options=types.HttpOptions(
                client_args={"transport": transport},
                async_client_args={"transport": transport},
            ),
        ) as client:
            kwargs = {
                "model": "gemini-2.5-flash",
                "contents": "What is the weather in Paris?",
                "config": (
                    config.model_dump(exclude_none=True)
                    if config_dict
                    else config
                ),
            }
            if asynchronous:
                async with client.aio as async_client:
                    if streaming:
                        stream = (
                            await async_client.models.generate_content_stream(
                                **kwargs
                            )
                        )
                        assert not span_exporter.get_finished_spans()
                        responses = [chunk async for chunk in stream]
                    else:
                        responses = [
                            await async_client.models.generate_content(
                                **kwargs
                            )
                        ]
            elif streaming:
                stream = client.models.generate_content_stream(**kwargs)
                assert not span_exporter.get_finished_spans()
                responses = list(stream)
            else:
                responses = [client.models.generate_content(**kwargs)]

    assert [result.text for result in responses] == ["Sunny."]
    assert declaration.model_dump(mode="json") == original_declaration
    (request,) = requests
    sent_declaration = types.FunctionDeclaration.model_validate(
        json.loads(request.content)["tools"][0]["functionDeclarations"][0]
    )
    assert sent_declaration.model_dump(mode="json") == original_declaration
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_RESPONSE_MODEL]
        == "gemini-2.5-flash"
    )
    if capture_content:
        definitions = span.attributes[
            gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS
        ]
        assert isinstance(definitions, str)
        assert json.loads(definitions) == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": expected_parameters,
            }
        ]
    else:
        assert gen_ai_attributes.GEN_AI_TOOL_DEFINITIONS not in span.attributes


def test_tool_definition_prefers_parameters() -> None:
    declaration = types.FunctionDeclaration(
        name="get_weather",
        parameters=_PARAMETERS,
        parameters_json_schema=_JSON_SCHEMA,
    )
    (definition,) = _tool_to_tool_definition(
        types.Tool(function_declarations=[declaration])
    )
    assert definition.parameters == _PARAMETERS
