"""Seven proposal agents; the Supervisor coordinates Identity, Order and Policy."""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from . import nodes
from .edges import next_agent
from .states import RetailState


def build_graph(adapter, agents, policy: str):
    builder = StateGraph(RetailState)
    dependencies = {"adapter": adapter, "agents": agents, "policy": policy}
    functions = {
        "supervisor": nodes.supervisor,
        "identity": nodes.identity_agent,
        "order": nodes.order_agent,
        "policy": nodes.policy_agent,
        "decision": nodes.decision_agent,
        "action_payment": nodes.action_payment_agent,
        "verification": nodes.verification_agent,
    }
    for name, function in functions.items():
        builder.add_node(name, partial(function, **dependencies))
    builder.add_edge(START, "supervisor")
    routes = {
        "supervisor": ["identity", "order", "policy", "decision", "end"],
        "identity": ["identity", "supervisor", "decision", "end"],
        "order": ["order", "supervisor", "decision", "end"],
        "decision": ["decision", "supervisor", "action_payment", "end"],
        "action_payment": ["verification", "end"],
    }
    for name, destinations in routes.items():
        builder.add_conditional_edges(
            name,
            next_agent,
            {target: END if target == "end" else target for target in destinations},
        )
    builder.add_edge("policy", "supervisor")
    builder.add_edge("verification", END)
    schemas = ["SupervisorPlan", "PendingAction", "ToolCall", "PolicyDecision"]
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[("retail_mas.objects", name) for name in schemas]
    )
    return builder.compile(checkpointer=InMemorySaver(serde=serializer))
