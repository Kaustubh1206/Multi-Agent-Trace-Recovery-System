"""Observed runtime state only; hidden task instructions/oracles do not belong here."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .objects import AgentStep, HandoffResult, PendingAction, PolicyDecision, SupervisorPlan


class RetailState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    source_task_id: str
    scenario_id: str
    customer_query: str
    plan: SupervisorPlan
    authenticated_user_id: str | None
    observations: list[dict[str, Any]]
    step: AgentStep | None
    active_agent: str
    next_node: str
    pending_action: PendingAction | None
    policy_result: PolicyDecision | None
    payment_info: dict[str, Any] | None
    confirmation: str | None
    execution_result: dict[str, Any] | None
    response: str
    error: str | None
    steps: int
    turns: int
    handoff: HandoffResult
