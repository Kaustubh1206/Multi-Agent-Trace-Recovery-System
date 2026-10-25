"""Evaluation trajectory of actual tool activity; disposable preflight is excluded."""

import json

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

from retail_mas.policy import IDENTITY_TOOLS, explicit_answer
from retail_mas.tools import RetailAdapter


class RecordedAdapter(RetailAdapter):
    def __init__(self, db):
        super().__init__(db)
        self.trajectory = []
        self.commits = []
        self.protected_reads = []
        self.rejections = []
        self.transfers = 0
        self.last_confirmation = None

    def record(self, name, arguments, result):
        call = ToolCall(id=f"tool-{len(self.trajectory)}", name=name, arguments=arguments)
        self.trajectory.extend(
            [
                AssistantMessage(role="assistant", tool_calls=[call]),
                ToolMessage(role="tool", id=call.id, content=json.dumps(result)),
            ]
        )

    def read(self, role, call):
        authenticated = bool(self.user_id)
        try:
            result = super().read(role, call)
        except (ValueError, TypeError) as error:
            # Refusals/lookup errors stay separate from successful native tool replay.
            self.rejections.append({"tool": call.name, "error": str(error)})
            raise
        if call.name in IDENTITY_TOOLS:
            raw = result["authenticated_user_id"]
        else:
            self.protected_reads.append(authenticated)
            raw = result
        self.record(call.name, call.arguments, raw)
        return result

    def confirm(self, fingerprint, customer_reply):
        confirmed = super().confirm(fingerprint, customer_reply)
        self.last_confirmation = (fingerprint, customer_reply) if confirmed else None
        return confirmed

    def execute(self, action):
        args = action.call.arguments
        order = self._tools.db.orders.get(args.get("order_id"))
        owned = bool(self.user_id) and (
            order.user_id == self.user_id if order else args.get("user_id") == self.user_id
        )
        confirmed = (
            self._approved == action.fingerprint
            and self.last_confirmation is not None
            and self.last_confirmation[0] == action.fingerprint
            and explicit_answer(self.last_confirmation[1], action.call.name) == "yes"
        )
        result = super().execute(action)
        self.commits.append(
            {
                "fingerprint": action.fingerprint,
                "order_id": args.get("order_id"),
                "user_id": self.user_id,
                "authenticated_owned": owned,
                "exact_confirmation": confirmed,
            }
        )
        self.record(action.call.name, args, result)
        return result

    def transfer(self, summary):
        result = super().transfer(summary)
        self.transfers += 1
        self.record("transfer_to_human_agents", {"summary": summary}, result)
        return result
