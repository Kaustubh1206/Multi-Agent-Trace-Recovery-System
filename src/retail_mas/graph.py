"""Compile the retail graph with real customer interrupts and a per-session checkpoint."""

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from . import nodes
from .edges import next_node
from .states import RetailState


def build_graph(adapter, agents, policy: str):
    builder = StateGraph(RetailState)
    dependencies = {"adapter": adapter, "agents": agents, "policy": policy}
    builder.add_node("supervisor", partial(nodes.supervisor, **dependencies))
    for role in ["identity", "order", "decision"]:
        builder.add_node(role, partial(nodes.specialist, role=role, **dependencies))
    builder.add_node("policy", partial(nodes.policy_agent, **dependencies))
    builder.add_node("payment", partial(nodes.payment_agent, **dependencies))
    builder.add_node("read", partial(nodes.read_tool, adapter=adapter))
    builder.add_node("prepare", partial(nodes.prepare_action, adapter=adapter))
    builder.add_node("ask", nodes.ask_customer)
    builder.add_node("confirm", partial(nodes.confirm_action, adapter=adapter))
    builder.add_node("execute", partial(nodes.execute_action, adapter=adapter))
    builder.add_node("transfer", partial(nodes.transfer_customer, adapter=adapter))
    builder.add_node("stop", nodes.respond)
    builder.add_edge(START, "supervisor")
    routes = {
        "supervisor": ["ask", "identity", "order", "decision", "stop"],
        "identity": ["read", "ask", "stop"],
        "order": ["read", "ask", "decision", "stop"],
        "decision": ["prepare", "ask", "order", "transfer", "stop"],
        "read": ["identity", "order"],
        "prepare": ["policy", "decision"],
        "policy": ["payment", "confirm", "stop"],
        "payment": ["confirm"],
        "ask": ["supervisor", "stop"],
        "confirm": ["execute", "stop", "supervisor"],
        "execute": ["stop"],
        "transfer": ["stop"],
    }
    for name, destinations in routes.items():
        builder.add_conditional_edges(name, next_node, {target: target for target in destinations})
    builder.add_edge("stop", END)
    # Explicitly permit our small schemas when resuming checkpoints; never enable pickle.
    schemas = [
        "SupervisorPlan",
        "AgentStep",
        "ToolCall",
        "PendingAction",
        "PolicyDecision",
        "HandoffResult",
    ]
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[("retail_mas.objects", name) for name in schemas]
    )
    return builder.compile(checkpointer=InMemorySaver(serde=serializer))
