"""Five LLM roles; execution and database verification stay inside Decision."""

from functools import partial
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from . import nodes
from .data_loader import COMMIT
from .edges import next_agent
from .states import CustomerInput, RetailState


def run_config(source_task_id: str | None = None):
    """Group customer turns and interrupt resumes into one LangSmith thread."""
    thread_id = str(uuid4())
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 512,
        "run_name": f"retail-task-{source_task_id}" if source_task_id else "retail-manual",
        "tags": ["retail-mas", "manual"],
        "metadata": {
            "thread_id": thread_id,
            "source_task_id": source_task_id or "manual",
            "execution_mode": "manual",
            "benchmark_commit": COMMIT,
        },
    }


def build_graph(adapter, agents, policy: str, *, server=False, wrap_node=None):
    """Use local checkpoints for the CLI; let Agent Server manage its own checkpoints."""
    builder = StateGraph(RetailState, input_schema=CustomerInput)
    dependencies = {"adapter": adapter, "agents": agents, "policy": policy}
    functions = {
        "supervisor": nodes.supervisor,
        "identity": nodes.identity_agent,
        "order": nodes.order_agent,
        "policy": nodes.policy_agent,
        "decision": nodes.decision_agent,
    }
    for name, function in functions.items():
        node = partial(function, **dependencies)
        builder.add_node(name, wrap_node(node) if wrap_node else node)
    builder.add_edge(START, "supervisor")
    routes = {
        "supervisor": ["identity", "order", "policy", "decision", "end"],
        "identity": ["identity", "order", "policy", "decision", "end"],
        "order": ["order", "policy", "decision", "end"],
        "decision": ["decision", "supervisor", "order", "end"],
    }
    for name, destinations in routes.items():
        builder.add_conditional_edges(
            name, next_agent,
            {target: END if target == "end" else target for target in destinations},
        )
    builder.add_edge("policy", "decision")
    schemas = ["SupervisorPlan", "PendingAction", "ToolCall", "PolicyDecision"]
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[("retail_mas.objects", name) for name in schemas]
    )
    checkpointer = None if server else InMemorySaver(serde=serializer)
    return builder.compile(checkpointer=checkpointer, name="RetailMAS")
