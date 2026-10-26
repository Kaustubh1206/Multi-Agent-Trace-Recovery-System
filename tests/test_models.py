import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from retail_mas.models import AgentModels, ModelOutputError, describe_model_error
from retail_mas.objects import PolicyDecision


class JSONOnlyModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("Provider has no native structured output")

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=next(self.outputs))


def test_validated_json_fallback():
    model = JSONOnlyModel(["not JSON", '{"allowed": false, "reason": "Not eligible"}'])
    result = AgentModels(model, output_method="json_schema").invoke(
        "policy", PolicyDecision, "Policy prompt", {}
    )
    assert not result.allowed
    assert model.calls == 2


def test_invalid_fallback_fails_closed():
    model = JSONOnlyModel(['{"allowed": true}', "invalid"])
    with pytest.raises(ValueError, match="Invalid structured output"):
        AgentModels(model, output_method="json_schema").invoke(
            "policy", PolicyDecision, "Policy prompt", {}
        )


def test_network_errors_are_not_retried_as_schema_failures():
    class FailingModel(JSONOnlyModel):
        def with_structured_output(self, *args, **kwargs):
            raise ConnectionError("Provider unavailable")

    model = FailingModel([])
    with pytest.raises(ConnectionError):
        AgentModels(model, output_method="json_schema").invoke(
            "policy", PolicyDecision, "Policy prompt", {}
        )
    assert model.calls == 0


def mock_chat(handler):
    """Real LangChain/SDK serialization, with every HTTP request intercepted locally."""
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        use_responses_api=False,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def completion(content):
    return httpx.Response(
        200,
        json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def test_groq_json_generation_rejection_uses_validated_fallback():
    import json

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Failed to generate JSON. Please adjust your prompt.",
                        "code": "json_validate_failed",
                        "type": "invalid_request_error",
                        "failed_generation": "not valid JSON",
                    }
                },
            )
        return completion('{"allowed": false, "reason": "Not eligible"}')

    result = AgentModels(mock_chat(handler), output_method="json_schema").invoke(
        "policy", PolicyDecision, "Policy prompt", {}
    )
    assert not result.allowed
    assert len(requests) == 2
    assert requests[0]["response_format"]["json_schema"]["strict"] is False
    assert "response_format" not in requests[1]


def test_json_mode_skips_native_schema_request():
    import json

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion('{"allowed": false, "reason": "Not eligible"}')

    result = AgentModels(mock_chat(handler), output_method="json").invoke(
        "policy", PolicyDecision, "Policy prompt", {}
    )
    assert not result.allowed
    assert len(requests) == 1
    assert "response_format" not in requests[0]
    assert "JSON object" in requests[0]["messages"][0]["content"]


def test_unrelated_bad_request_does_not_trigger_schema_fallback():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Unsupported temperature",
                    "param": "temperature",
                    "code": "invalid_value",
                }
            },
        )

    with pytest.raises(BadRequestError):
        AgentModels(mock_chat(handler), output_method="json_schema").invoke(
            "policy", PolicyDecision, "Policy prompt", {}
        )
    assert len(requests) == 1


def test_diagnostics_show_cause_without_secrets_or_generated_content(monkeypatch):
    secret = "gsk_private-test-key"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
    error = BadRequestError(
        "Raw response should not be printed",
        response=response,
        body={
            "error": {
                "message": f"Invalid response_format; key {secret}; https://example.com/?token=hidden",
                "code": "json_validate_failed",
                "param": "response_format",
                "failed_generation": "private-customer-text",
            },
        },
    )
    diagnostic = describe_model_error(error)
    assert "HTTP 400" in diagnostic
    assert "json_validate_failed" in diagnostic
    assert "response_format" in diagnostic
    assert secret not in diagnostic
    assert "token=hidden" not in diagnostic
    assert "private-customer-text" not in diagnostic


def test_invalid_output_method_fails_before_model_request():
    with pytest.raises(ValueError, match="RETAIL_OUTPUT_METHOD"):
        AgentModels(JSONOnlyModel([]), output_method="unsupported")


def test_validation_failure_has_a_readable_safe_message():
    error = ModelOutputError("Invalid structured output from supervisor; no action taken")
    assert describe_model_error(error) == str(error)


def test_openai_configuration_sends_native_model_id_and_compatible_parameters(monkeypatch):
    import json

    from retail_mas import models

    requests = []

    def handler(request):
        requests.append((str(request.url), json.loads(request.content)))
        return completion('{"allowed": false, "reason": "Not eligible"}')

    def configured_client(**kwargs):
        return ChatOpenAI(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("RETAIL_MODEL", raising=False)
    monkeypatch.setenv("RETAIL_REASONING_EFFORT", "none")
    monkeypatch.setattr(models, "ChatOpenAI", configured_client)
    result = models.AgentModels(output_method="json").invoke(
        "policy", PolicyDecision, "Policy prompt", {}
    )
    assert not result.allowed
    endpoint, body = requests[0]
    assert endpoint == "https://api.openai.com/v1/chat/completions"
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning_effort"] == "none"
    assert "temperature" not in body


def test_native_structured_output_preserves_real_sdk_usage():
    requests = []

    def handler(request):
        requests.append(request)
        payload = completion('{"allowed": true, "reason": "Eligible"}').json()
        payload["usage"] = {
            "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        return httpx.Response(200, json=payload)

    agents = AgentModels(mock_chat(handler), output_method="json_schema")
    assert agents.invoke("policy", PolicyDecision, "Policy prompt", {}).allowed
    assert len(requests) == agents.usage["calls"] == 1
    assert agents.usage["responses_with_usage"] == 1
    assert agents.usage["input_tokens"] == 10
    assert agents.usage["output_tokens"] == 3
    assert agents.usage["cached_input_tokens"] == 4
