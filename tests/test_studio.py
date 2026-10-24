"""Offline Agent Server lifecycle checks, using real retail tools and scripted agents."""

import asyncio
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from retail_mas import studio

from .support import ScriptedAgents, cancellation_steps

EXECUTION = SimpleNamespace(execution_runtime=True)
QUERY = {
    "customer_query": "Emma Smith ZIP 10192, no email. Cancel #W2417020: no longer needed.",
    "messages": [{"role": "human", "content": "Cancel my order #W2417020."}],
    "new_request": True,
}


@pytest.fixture
def sessions(monkeypatch):
    monkeypatch.setattr(studio, "_sessions", {})
    monkeypatch.setattr(studio, "AgentModels", lambda: ScriptedAgents(cancellation_steps()))
    return studio._sessions


def graph_for(thread_id, saver):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 512}
    graph = asyncio.run(studio.make_graph(config, EXECUTION))
    assert graph.checkpointer is None  # Agent Server supplies persistence itself.
    graph.checkpointer = saver
    return graph, config


def test_schema_rendering_needs_no_credentials_or_dataset(monkeypatch, sessions):
    def unavailable():
        pytest.fail("Schema rendering must not create a retail session")

    monkeypatch.setattr(studio, "RetailDataset", unavailable)
    graph = asyncio.run(studio.make_graph({}, SimpleNamespace(execution_runtime=None)))
    assert len(graph.nodes) == 8  # START plus the seven proposal roles.
    fields = graph.get_input_jsonschema()["properties"]
    assert "customer_query" in fields
    assert "confirmation" not in fields
    assert "studio_session_id" not in fields
    assert not sessions


def test_factory_reuses_session_on_confirmation_and_isolates_threads(sessions):
    saver = InMemorySaver()
    graph, config = graph_for("first", saver)
    result = graph.invoke(QUERY, config)
    first = sessions["first"]
    assert result["__interrupt__"][0].value["kind"] == "confirmation"
    assert isinstance(result["pending_action"], dict)
    assert first.adapter.write_count == 0

    graph, config = graph_for("first", saver)
    result = graph.invoke(Command(resume="yes"), config)
    assert sessions["first"] is first
    assert result["execution_result"]["status"] == "cancelled"
    assert first.adapter.write_count == 1
    assert not first.agents.steps

    graph, config = graph_for("second", saver)
    second = sessions["second"]
    assert second.adapter.user_id is None
    assert second.adapter.snapshot().orders["#W2417020"].status == "pending"
    assert second.token != first.token
    graph.invoke(QUERY, config)
    graph, config = graph_for("second", saver)
    graph.invoke(Command(resume="no"), config)
    assert second.adapter.write_count == 0
    assert second.adapter.snapshot().orders["#W2417020"].status == "pending"


def test_checkpoint_from_old_process_cannot_resume_with_fresh_database(sessions):
    saver = InMemorySaver()
    graph, config = graph_for("expired", saver)
    graph.invoke(QUERY, config)
    sessions.clear()  # Mimic a server restart while its checkpoints remain on disk.
    graph, config = graph_for("expired", saver)
    with pytest.raises(ValueError, match="Create a new Studio thread"):
        graph.invoke(Command(resume="yes"), config)
    assert sessions["expired"].adapter.write_count == 0
    assert sessions["expired"].adapter.user_id is None


def test_execution_requires_a_thread(sessions):
    with pytest.raises(ValueError, match="Create a Studio thread"):
        asyncio.run(studio.make_graph({}, EXECUTION))
    assert not sessions
