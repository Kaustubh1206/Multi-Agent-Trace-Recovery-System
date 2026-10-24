"""Small validated outputs shared by graph nodes."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(Output):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentStep(Output):
    kind: Literal["tool", "ask", "complete", "respond", "propose", "transfer"]
    text: str = ""
    tool: ToolCall | None = None


class SupervisorPlan(Output):
    intent: Literal["inquiry", "change", "unsupported"]
    purpose: str
    question: str = ""


class PolicyDecision(Output):
    allowed: bool
    reason: str
    constraints: list[str] = Field(default_factory=list)


class PaymentFinding(Output):
    reason: str
    customer_explanation: str


class VerificationFinding(Output):
    verified: bool
    reason: str


class PendingAction(Output):
    call: ToolCall
    snapshot_hash: str
    expected_db_hash: str
    financial_effect: dict[str, Any] = Field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        # Binds approval to exact arguments AND the state used for the proposal.
        value = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(value.encode()).hexdigest()
