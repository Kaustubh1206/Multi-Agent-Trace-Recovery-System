import pytest
from langchain_core.messages import AIMessage

from retail_mas.models import AgentModels
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
    result = AgentModels(model).invoke("policy", PolicyDecision, "Policy prompt", {})
    assert not result.allowed
    assert model.calls == 2


def test_invalid_fallback_fails_closed():
    model = JSONOnlyModel(['{"allowed": true}', "invalid"])
    with pytest.raises(ValueError, match="Invalid structured output"):
        AgentModels(model).invoke("policy", PolicyDecision, "Policy prompt", {})


def test_network_errors_are_not_retried_as_schema_failures():
    class FailingModel(JSONOnlyModel):
        def with_structured_output(self, *args, **kwargs):
            raise ConnectionError("Provider unavailable")

    model = FailingModel([])
    with pytest.raises(ConnectionError):
        AgentModels(model).invoke("policy", PolicyDecision, "Policy prompt", {})
    assert model.calls == 0
