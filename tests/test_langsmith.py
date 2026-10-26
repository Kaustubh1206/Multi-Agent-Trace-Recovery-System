"""Tracing checks stay local; no LangSmith key, uploads or paid model calls."""

import json

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langgraph.types import Command
from langsmith import Client, tracing_context

from retail_mas.graph import build_graph, run_config
from retail_mas.tools import RetailAdapter

from .support import ScriptedAgents, authenticate_emma, call, cancellation, cancellation_steps


@pytest.fixture
def adapter(dataset):
    return RetailAdapter(dataset.fresh_db())


@pytest.fixture
def local_client(monkeypatch):
    client = Client(api_url="https://traces.invalid", api_key="test-key", auto_batch_tracing=False)

    def no_upload(*args, **kwargs):
        pytest.fail("Offline tracing tests must not upload")

    monkeypatch.setattr(Client, "create_run", no_upload)
    monkeypatch.setattr(Client, "update_run", no_upload)
    return client


def test_graph_traces_group_confirmation_resumes_and_name_all_agents(adapter, dataset):
    collector = RunCollectorCallbackHandler()
    config = run_config("69")
    config["callbacks"] = [collector]
    graph = build_graph(adapter, ScriptedAgents(cancellation_steps()), dataset.policy)
    query = "Emma Smith ZIP 10192, no email. Cancel #W2417020, no longer needed."
    paused = graph.invoke(
        {"customer_query": query, "messages": [HumanMessage(content=query)]}, config
    )
    assert paused["__interrupt__"]
    assert adapter.write_count == 0
    result = graph.invoke(Command(resume="yes"), config)
    assert result["verification_result"]["verified"]
    assert adapter.write_count == 1
    assert len(collector.traced_runs) == 2
    names = set()
    for root in collector.traced_runs:
        assert root.name == "retail-task-69"
        assert root.extra["metadata"]["thread_id"] == config["configurable"]["thread_id"]
        assert root.extra["metadata"]["source_task_id"] == "69"
        for child in root.child_runs:
            names.add(child.name)
            assert child.extra["metadata"]["thread_id"] == config["metadata"]["thread_id"]
    assert {
        "supervisor",
        "identity",
        "order",
        "policy",
        "decision",
    } <= names
    assert run_config("69")["metadata"]["thread_id"] != config["metadata"]["thread_id"]


def test_tool_trace_logs_scoped_call_not_the_adapter_or_fixture(adapter, local_client):
    runs = []
    with tracing_context(
        enabled="local", client=local_client, metadata={"thread_id": "test-thread"}
    ):
        result = adapter.read(
            "identity",
            call("find_user_id_by_email", email="emma.smith3991@example.com"),
            langsmith_extra={"on_end": runs.append},
        )
    assert result["authenticated_user_id"] == "emma_smith_8564"
    assert runs[0].name == "retail.read"
    assert runs[0].inputs == {
        "role": "identity",
        "tool": "find_user_id_by_email",
        "arguments": {"email": "emma.smith3991@example.com"},
    }
    assert runs[0].extra["metadata"]["thread_id"] == "test-thread"
    serialized = json.dumps(runs[0].inputs)
    for forbidden in ["self", "users", "products", "evaluation_criteria", "user_scenario"]:
        assert forbidden not in serialized


def test_tracing_cannot_bypass_confirmation_guard(adapter, local_client):
    authenticate_emma(adapter)
    action = adapter.prepare(cancellation())
    runs = []
    with tracing_context(enabled="local", client=local_client):
        with pytest.raises(ValueError, match="approval"):
            adapter.execute(action, langsmith_extra={"on_end": runs.append})
    assert adapter.write_count == 0
    assert runs[0].name == "retail.execute"
    assert "approval" in runs[0].error
    assert "expected_db_hash" not in runs[0].inputs


def test_tracing_observes_one_commit_and_actual_verification(adapter, local_client):
    authenticate_emma(adapter)
    action = adapter.prepare(cancellation())
    assert adapter.confirm(action.fingerprint, "yes")
    runs = []
    with tracing_context(enabled="local", client=local_client):
        result = adapter.execute(action, langsmith_extra={"on_end": runs.append})
        checked = adapter.verify(action, result, langsmith_extra={"on_end": runs.append})
    assert checked["matches_confirmed_action"]
    assert adapter.write_count == 1
    assert [run.name for run in runs] == ["retail.execute", "retail.verify"]
    assert runs[0].inputs["arguments"] == cancellation().arguments
    assert runs[1].child_runs[0].name == "retail.read"


def test_disabled_tracing_does_not_capture_or_upload(adapter, local_client):
    runs = []
    with tracing_context(enabled=False, client=local_client):
        result = adapter.read(
            "identity",
            call("find_user_id_by_email", email="emma.smith3991@example.com"),
            langsmith_extra={"on_end": runs.append},
        )
    assert result["authenticated_user_id"] == "emma_smith_8564"
    assert runs == []
