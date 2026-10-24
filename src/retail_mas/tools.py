"""Serial, authenticated adapter around unchanged official RetailTools APIs."""

import hashlib
import inspect
import json
from threading import RLock
from typing import get_type_hints

from pydantic import BaseModel, TypeAdapter
from tau2.domains.retail.tools import RetailTools

from .objects import PendingAction, ToolCall
from .policy import IDENTITY_TOOLS, ORDER_TOOLS, check_action, explicit_answer

READ_SCOPES = {
    "identity": IDENTITY_TOOLS,
    "order": ORDER_TOOLS,
    "action_payment": {"get_user_details"},
    "verification": {"get_user_details", "get_order_details"},
}


def plain(value):
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def database_hash(db):
    return hashlib.sha256(json.dumps(plain(db), sort_keys=True).encode()).hexdigest()


class RetailAdapter:
    """One instance per conversation/trial; DB, authentication and consent are private."""

    def __init__(self, db):
        self._tools = RetailTools(db.model_copy(deep=True))
        self._user_id: str | None = None
        self._approved: str | None = None
        self._prepared: PendingAction | None = None
        self._committed: set[str] = set()
        self._lock = RLock()
        self.write_count = 0

    @property
    def user_id(self):
        return self._user_id

    def snapshot(self):
        """Independent copy for the test harness, never attached to agent state."""
        with self._lock:
            return self._tools.db.model_copy(deep=True)

    def _validate_call(self, call: ToolCall):
        if call.name not in self._tools.tools:
            raise ValueError("Unknown retail tool")
        function = self._tools.tools[call.name]
        inspect.signature(function).bind(**call.arguments)
        hints = get_type_hints(function)
        for name, value in call.arguments.items():
            TypeAdapter(hints[name]).validate_python(value, strict=True)

    def schemas(self, names: set[str]) -> list[dict]:
        tools = self._tools.get_tools(include=sorted(names))
        return [tools[name].openai_schema for name in sorted(tools)]

    def read(self, role: str, call: ToolCall):
        with self._lock:
            allowed = READ_SCOPES.get(role, set())
            if call.name not in allowed:
                raise ValueError("Tool outside this specialist's read scope")
            self._validate_call(call)
            if call.name in IDENTITY_TOOLS:
                result = self._tools.use_tool(call.name, **call.arguments)
                if self._user_id and result != self._user_id:
                    raise ValueError("Only one customer can be served per conversation")
                self._user_id = result
                return {"authenticated_user_id": result}
            if not self._user_id:
                raise ValueError("Authenticate before protected activity")
            args = call.arguments
            if call.name == "get_user_details" and args.get("user_id") != self._user_id:
                raise ValueError("Cannot reveal another customer's information")
            if call.name == "get_order_details":
                order = self._tools.db.orders.get(args.get("order_id"))
                if order is None or order.user_id != self._user_id:
                    raise ValueError(
                        "Order unavailable for this customer; please check the reference"
                    )
            return plain(self._tools.use_tool(call.name, **args))

    def _snapshot_hash(self, call: ToolCall) -> str:
        db = self._tools.db
        values = {"user": plain(db.users[self._user_id])}
        order_id = call.arguments.get("order_id")
        if order_id:
            values["order"] = plain(db.orders[order_id])
            values["products"] = {
                item.product_id: plain(db.products[item.product_id])
                for item in db.orders[order_id].items
            }
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def prepare(self, call: ToolCall) -> PendingAction:
        with self._lock:
            self.invalidate_consent()
            self._validate_call(call)
            check_action(self._tools.db, self._user_id, call)
            # Official APIs enforce payment balance, return method, item quantities, etc.
            # Validate on a disposable copy; this is not a customer action or benchmark run.
            trial = RetailTools(self._tools.db.model_copy(deep=True))
            trial.use_tool(call.name, **call.arguments)
            effect = self._financial_effect(call, trial.db)
            action = PendingAction(
                call=call.model_copy(deep=True),
                snapshot_hash=self._snapshot_hash(call),
                expected_db_hash=database_hash(trial.db),
                financial_effect=effect,
            )
            self._prepared = action.model_copy(deep=True)
            return action

    def _financial_effect(self, call, after):
        args = call.arguments
        if "order_id" not in args:
            return {"amount": 0, "note": "Default profile address only"}
        before = self._tools.db.orders[args["order_id"]]
        updated = after.orders[args["order_id"]]
        name = call.name
        if name == "cancel_pending_order":
            return {
                "refunds": [
                    plain(p) for p in updated.payment_history[len(before.payment_history) :]
                ],
                "timing": "Gift card immediately; other methods 5–7 business days",
            }
        if name == "return_delivered_order_items":
            amount = sum(
                next(i.price for i in before.items if i.item_id == item)
                for item in args["item_ids"]
            )
            return {
                "refund_after_return": round(amount, 2),
                "payment_method_id": args["payment_method_id"],
                "note": "Return requested; follow-up return instructions by email",
            }
        if name == "exchange_delivered_order_items":
            return {
                "price_difference": updated.exchange_price_difference,
                "payment_method_id": args["payment_method_id"],
                "note": "Exchange requested; follow-up instructions by email; no new order needed",
            }
        return {
            "transactions": [
                plain(p) for p in updated.payment_history[len(before.payment_history) :]
            ],
            "timing": "Gift card refund immediately; other refunds 5–7 business days",
        }

    def invalidate_consent(self):
        self._approved = None
        self._prepared = None

    def confirm(self, fingerprint: str, customer_reply: str) -> bool:
        with self._lock:
            self._approved = None
            if (
                self._prepared
                and fingerprint == self._prepared.fingerprint
                and explicit_answer(customer_reply) == "yes"
            ):
                self._approved = fingerprint
                return True
            return False

    def execute(self, action: PendingAction):
        with self._lock:
            fingerprint = action.fingerprint
            if fingerprint in self._committed:
                raise ValueError("Action already committed")
            if (
                not self._prepared
                or self._prepared.fingerprint != fingerprint
                or self._approved != fingerprint
            ):
                raise ValueError("No explicit customer approval for this exact action")
            # Consume consent before any execution, including a failed execution.
            self.invalidate_consent()
            self._validate_call(action.call)
            check_action(self._tools.db, self._user_id, action.call)
            if self._snapshot_hash(action.call) != action.snapshot_hash:
                raise ValueError("Relevant data changed; prepare and confirm a new action")
            # Atomic in-memory commit: official tools can otherwise partially mutate on error.
            trial = RetailTools(self._tools.db.model_copy(deep=True))
            result = plain(trial.use_tool(action.call.name, **action.call.arguments))
            self._tools = trial
            self._committed.add(fingerprint)
            self.write_count += 1
            return result

    def transfer(self, summary: str):
        with self._lock:
            return self._tools.transfer_to_human_agents(summary=summary)

    def verify(self, action: PendingAction, execution_result: dict) -> dict:
        """Compare actual DB with this customer's preflighted action, not task labels."""
        with self._lock:
            args = action.call.arguments
            target = (
                {"order_id": args["order_id"]}
                if "order_id" in args
                else {"user_id": args["user_id"]}
            )
            name = "get_order_details" if "order_id" in args else "get_user_details"
            observed = self.read("verification", ToolCall(name=name, arguments=target))
            matches = (
                observed == execution_result
                and database_hash(self._tools.db) == action.expected_db_hash
            )
            return {"matches_confirmed_action": matches, "record": observed}
