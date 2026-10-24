"""Shared observed facts and the current customer request; no benchmark oracle."""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .objects import PendingAction, PolicyDecision, SupervisorPlan


class CustomerInput(TypedDict, total=False):
    """Public request fields; internal approvals and routing are never input fields."""

    messages: Annotated[list[AnyMessage], add_messages]
    source_task_id: str
    scenario_id: str
    customer_query: str
    new_request: bool


class RetailState(CustomerInput, total=False):
    studio_session_id: str
    plan: SupervisorPlan
    authenticated_user_id: str | None
    observations: list[dict[str, Any]]
    order_complete: bool
    next_agent: str
    handoff_request: str
    policy_result: PolicyDecision | None
    pending_action: PendingAction | None
    awaiting_customer: Literal["customer", "confirmation"] | None
    question: str
    confirmation: str | None
    payment_info: dict[str, Any] | None
    execution_result: dict[str, Any] | None
    verification_result: dict[str, Any] | None
    response: str
    error: str | None
    steps: int
    turns: int
