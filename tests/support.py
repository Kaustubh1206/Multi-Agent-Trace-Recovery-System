"""Scripted model outputs are test doubles, not a production/offline 'LLM'."""

from collections import deque
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from retail_mas.graph import build_graph
from retail_mas.objects import ToolCall


def call(name, **arguments):
    return ToolCall(name=name, arguments=arguments)


def tool(name, **arguments):
    return {"kind": "tool", "tool": call(name, **arguments).model_dump()}


def proposal(name, **arguments):
    return {"kind": "propose", "tool": call(name, **arguments).model_dump()}


def plan(intent="change"):
    return ("supervisor", {"intent": intent, "purpose": "Handle the customer's current request"})


def reviews():
    return [
        ("policy", {"allowed": True, "reason": "Validated status, ownership and parameters"}),
        (
            "payment",
            {
                "reason": "Official tool preconditions passed",
                "customer_explanation": "Use the validated financial consequences",
            },
        ),
    ]


class ScriptedAgents:
    def __init__(self, steps):
        self.steps = deque(steps)
        self.contexts = []

    def invoke(self, role, schema, system, context):
        expected_role, output = self.steps.popleft()
        assert expected_role == role
        self.contexts.append((role, system, context))
        return schema.model_validate(output)


def session(adapter, dataset, steps, query, source_task_id=None):
    agents = ScriptedAgents(steps)
    graph = build_graph(adapter, agents, dataset.policy)
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 512}
    update = {"messages": [HumanMessage(content=query)], "customer_query": query}
    if source_task_id:
        update.update(source_task_id=source_task_id, scenario_id="offline-" + source_task_id)
    return graph, agents, config, graph.invoke(update, config)


def resume(graph, config, reply):
    return graph.invoke(Command(resume=reply), config)


def authenticate_emma(adapter):
    adapter.read(
        "identity",
        call("find_user_id_by_name_zip", first_name="Emma", last_name="Smith", zip="10192"),
    )


def cancellation():
    return call("cancel_pending_order", order_id="#W2417020", reason="no longer needed")


def cancellation_steps():
    return [
        plan(),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Emma", last_name="Smith", zip="10192"),
        ),
        ("order", tool("get_order_details", order_id="#W2417020")),
        ("order", {"kind": "complete"}),
        ("decision", {"kind": "propose", "tool": cancellation().model_dump()}),
        *reviews(),
    ]
