"""Deterministic policy boundaries, independent of model judgments."""

from collections import Counter

from .objects import ToolCall

WRITE_TOOLS = {
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_pending_order_items",
    "modify_user_address",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}
IDENTITY_TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
ORDER_TOOLS = {
    "get_order_details",
    "get_user_details",
    "get_product_details",
    "get_item_details",
    "list_all_product_types",
    "calculate",
}


def explicit_answer(reply: str) -> str:
    """Conservative consent: changes embedded in a 'yes' are NOT approval."""
    normalized = reply.strip().lower().rstrip(".! ")
    if normalized in {"yes", "yes, confirm", "i confirm"}:
        return "yes"
    if normalized in {"no", "cancel", "do not proceed"}:
        return "no"
    return "unclear"


def check_action(db, user_id: str | None, call: ToolCall) -> None:
    """Extra policy rules not consistently enforced by the official tools."""
    if not user_id:
        raise ValueError("Authenticate by email or name + ZIP first")
    if call.name not in WRITE_TOOLS:
        raise ValueError("Unsupported retail write")
    args = call.arguments
    if call.name == "modify_user_address":
        if args.get("user_id") != user_id:
            raise ValueError("Cannot act for another customer")
        return
    order = db.orders.get(args.get("order_id"))
    if order is None or order.user_id != user_id:
        raise ValueError("Order unavailable for this customer; please check the reference")
    required_status = "delivered" if "delivered" in call.name else "pending"
    if order.status != required_status:
        raise ValueError(f"This action requires an order with status '{required_status}'")
    if "item_ids" not in args:
        return
    item_ids = args["item_ids"]
    counts = Counter(item_ids)
    available = Counter(item.item_id for item in order.items)
    if not item_ids or counts - available:
        raise ValueError("Select a nonempty list of actual order item IDs with valid quantities")
    if "new_item_ids" not in args:
        return
    new_ids = args["new_item_ids"]
    if len(new_ids) != len(item_ids):
        raise ValueError("Each changed item needs one replacement")
    variants = []
    for old_id, new_id in zip(item_ids, new_ids):
        item = next(item for item in order.items if item.item_id == old_id)
        variant = db.products[item.product_id].variants.get(new_id)
        if variant is None or not variant.available:
            raise ValueError("Replacement must be an available variant of the same product")
        if old_id == new_id or variant.options == item.options:
            raise ValueError("Replacement must have different product options")
        variants.append(variant)
    if call.name == "modify_pending_order_items":
        # Pinned upstream reuses the final loop's variant during mutation.
        # Also avoid overlapping replacements that upstream resolves sequentially.
        if len({(v.price, tuple(sorted(v.options.items()))) for v in variants}) > 1:
            raise ValueError(
                "Pinned upstream multi-item modification defect: human assistance needed"
            )
        if set(item_ids) & set(new_ids):
            raise ValueError("Overlapping pending replacements need human assistance")


def confirmation_summary(action) -> str:
    """The exact canonical call is always visible, even if an LLM omits a detail."""
    import json

    name = action.call.name
    details = json.dumps(action.call.arguments, sort_keys=True)
    finance = json.dumps(action.financial_effect, sort_keys=True)
    note = ""
    if name in {"modify_pending_order_items", "exchange_delivered_order_items"}:
        note = " Confirm this includes ALL items you want changed; this operation is allowed once."
    if name == "modify_pending_order_items":
        note += " Further modifications/cancellation will not be available afterwards."
    return f"Proposed {name}: {details}\nFinancial consequences: {finance}.{note}\nProceed? Reply yes or no."
