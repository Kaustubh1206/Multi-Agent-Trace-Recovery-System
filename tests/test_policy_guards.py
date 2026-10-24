import pytest

from retail_mas.objects import PendingAction
from retail_mas.policy import explicit_answer

from .support import authenticate_emma, call, cancellation


def test_unauthenticated_cannot_read_or_write(adapter):
    with pytest.raises(ValueError, match="Authenticate"):
        adapter.read("order", call("get_order_details", order_id="#W2417020"))
    with pytest.raises(ValueError, match="Authenticate"):
        adapter.prepare(cancellation())
    assert adapter.write_count == 0


def test_cancellation_exact_consent_and_single_commit(adapter):
    authenticate_emma(adapter)
    initial = adapter.snapshot()
    action = adapter.prepare(cancellation())
    assert adapter.snapshot() == initial
    with pytest.raises(ValueError, match="approval"):
        adapter.execute(action)
    for reply in [
        "no",
        "maybe",
        "yes, but cancel a different order",
        "",
        "yes please change the reason",
    ]:
        assert not adapter.confirm(action.fingerprint, reply)
        with pytest.raises(ValueError):
            adapter.execute(action)
    assert adapter.confirm(action.fingerprint, "yes")
    result = adapter.execute(action)
    assert result["status"] == "cancelled"
    assert adapter.write_count == 1
    with pytest.raises(ValueError, match="already committed"):
        adapter.execute(action)
    after = adapter.snapshot()
    assert after.users["emma_smith_8564"].payment_methods["gift_card_8541487"].balance == 2736.4
    assert {k: v for k, v in after.orders.items() if k != "#W2417020"} == {
        k: v for k, v in initial.orders.items() if k != "#W2417020"
    }
    assert {k: v for k, v in after.users.items() if k != "emma_smith_8564"} == {
        k: v for k, v in initial.users.items() if k != "emma_smith_8564"
    }
    assert after.products == initial.products


def test_supplied_user_id_is_not_authentication(adapter):
    with pytest.raises(ValueError):
        adapter.read("identity", call("get_user_details", user_id="emma_smith_8564"))
    assert adapter.user_id is None


def test_wrong_customer_and_single_user_session(adapter):
    authenticate_emma(adapter)
    with pytest.raises(ValueError, match="unavailable"):
        adapter.read("order", call("get_order_details", order_id="#W2378156"))
    with pytest.raises(ValueError, match="unavailable"):
        adapter.prepare(
            call("cancel_pending_order", order_id="#W2378156", reason="no longer needed")
        )
    with pytest.raises(ValueError, match="another"):
        adapter.read("order", call("get_user_details", user_id="yusuf_rossi_9620"))
    with pytest.raises(ValueError, match="one customer"):
        adapter.read("identity", call("find_user_id_by_email", email="yusuf.rossi7301@example.com"))
    assert adapter.user_id == "emma_smith_8564"


def test_tool_scope_and_schema_validation(adapter):
    authenticate_emma(adapter)
    with pytest.raises(ValueError, match="scope"):
        adapter.read("order", cancellation())
    with pytest.raises((ValueError, TypeError)):
        adapter.prepare(call("cancel_pending_order", order_id=123, reason="no longer needed"))
    with pytest.raises((ValueError, TypeError)):
        adapter.prepare(
            call(
                "cancel_pending_order",
                order_id="#W2417020",
                reason="no longer needed",
                consent=True,
            )
        )
    with pytest.raises(ValueError):
        adapter.prepare(
            call("cancel_pending_order", order_id="#W2417020", reason="cheaper elsewhere")
        )


def test_tampered_action_and_fingerprint_rejected(adapter):
    authenticate_emma(adapter)
    action = adapter.prepare(cancellation())
    assert not adapter.confirm("wrong-fingerprint", "yes")
    assert adapter.confirm(action.fingerprint, "yes")
    changed = PendingAction.model_validate(action.model_dump())
    changed.call.arguments["reason"] = "ordered by mistake"
    with pytest.raises(ValueError, match="approval"):
        adapter.execute(changed)
    assert adapter.write_count == 0


def test_stale_order_requires_new_confirmation(adapter):
    authenticate_emma(adapter)
    action = adapter.prepare(cancellation())
    adapter.confirm(action.fingerprint, "yes")
    # Model an external environment change; no injection/recovery machinery.
    adapter._tools.db.orders["#W2417020"].address.city = "Changed externally"
    with pytest.raises(ValueError, match="changed"):
        adapter.execute(action)
    assert adapter.write_count == 0
    with pytest.raises(ValueError, match="approval"):
        adapter.execute(action)


@pytest.mark.parametrize("reply", ["yesterday", "yes, but a different item", "sure", "True", "1"])
def test_ambiguous_consent_is_not_yes(reply):
    assert explicit_answer(reply) == "unclear"


@pytest.mark.parametrize(
    "args",
    [
        {"item_ids": ["1656367028"], "new_item_ids": ["7706410293"]},  # product, not item ID
        {"item_ids": ["1151293680"], "new_item_ids": ["7747408585"]},  # other product
        {"item_ids": ["1151293680"], "new_item_ids": ["1151293680"]},  # same variant
        {"item_ids": [], "new_item_ids": []},
        {"item_ids": ["1151293680", "1151293680"], "new_item_ids": ["7706410293", "7706410293"]},
    ],
)
def test_invalid_item_identity_quantity_or_variant(adapter, args):
    adapter.read("identity", call("find_user_id_by_email", email="yusuf.rossi7301@example.com"))
    with pytest.raises(ValueError):
        adapter.prepare(
            call(
                "exchange_delivered_order_items",
                order_id="#W2378156",
                payment_method_id="credit_card_9513926",
                **args,
            )
        )
    assert adapter.write_count == 0


def test_unavailable_option(adapter):
    adapter.read("identity", call("find_user_id_by_email", email="yusuf.rossi7301@example.com"))
    product = adapter.snapshot().products["1656367028"]
    unavailable = next(v.item_id for v in product.variants.values() if not v.available)
    with pytest.raises(ValueError, match="available"):
        adapter.prepare(
            call(
                "exchange_delivered_order_items",
                order_id="#W2378156",
                item_ids=["1151293680"],
                new_item_ids=[unavailable],
                payment_method_id="credit_card_9513926",
            )
        )


def test_disallowed_return_payment(adapter):
    adapter.read("identity", call("find_user_id_by_email", email="chen.silva2698@example.com"))
    with pytest.raises(ValueError, match="original"):
        adapter.prepare(
            call(
                "return_delivered_order_items",
                order_id="#W9571698",
                item_ids=["6065192424"],
                payment_method_id="credit_card_1565124",
            )
        )


def test_insufficient_gift_card_and_wrong_status(adapter):
    authenticate_emma(adapter)
    original = adapter.snapshot()
    with pytest.raises(ValueError, match="balance"):
        # This delivered laptop has costlier available variants in the real fixture.
        order = original.orders["#W5605613"]
        old = order.items[0]
        variant = next(
            v
            for v in original.products[old.product_id].variants.values()
            if v.available and v.price > old.price + 62
        )
        adapter.prepare(
            call(
                "exchange_delivered_order_items",
                order_id=order.order_id,
                item_ids=[old.item_id],
                new_item_ids=[variant.item_id],
                payment_method_id="gift_card_8541487",
            )
        )
    with pytest.raises(ValueError, match="delivered"):
        adapter.prepare(
            call(
                "return_delivered_order_items",
                order_id="#W2417020",
                item_ids=["8997785118"],
                payment_method_id="gift_card_8541487",
            )
        )
    assert adapter.snapshot() == original


@pytest.mark.parametrize(
    "name",
    [
        "modify_pending_order_address",
        "modify_user_address",
        "modify_pending_order_payment",
        "modify_pending_order_items",
    ],
)
def test_other_official_write_apis_are_guarded_and_runnable(adapter, name):
    authenticate_emma(adapter)
    db = adapter.snapshot()
    if name.endswith("address"):
        address = db.users["emma_smith_8564"].address.model_dump()
        address["address1"] = "10 Example Lane"
        target = (
            {"user_id": "emma_smith_8564"}
            if name == "modify_user_address"
            else {"order_id": "#W2417020"}
        )
        action_call = call(name, **target, **address)
    elif name == "modify_pending_order_payment":
        action_call = call(name, order_id="#W2417020", payment_method_id="paypal_6228291")
    else:
        old = db.orders["#W2417020"].items[0]
        variant = next(
            v
            for v in db.products[old.product_id].variants.values()
            if v.available and v.item_id != old.item_id and v.options != old.options
        )
        action_call = call(
            name,
            order_id="#W2417020",
            item_ids=[old.item_id],
            new_item_ids=[variant.item_id],
            payment_method_id="paypal_6228291",
        )
    action = adapter.prepare(action_call)
    assert adapter.snapshot() == db
    with pytest.raises(ValueError, match="approval"):
        adapter.execute(action)
    assert adapter.confirm(action.fingerprint, "yes")
    adapter.execute(action)
    assert adapter.write_count == 1
    assert adapter.snapshot() != db


def test_known_upstream_multi_item_modification_defect_fails_closed(adapter):
    db = adapter.snapshot()
    candidates = None
    for order in db.orders.values():
        if order.status != "pending":
            continue
        replacements = []
        for item in order.items:
            variant = next(
                (
                    v
                    for v in db.products[item.product_id].variants.values()
                    if v.available and v.options != item.options
                ),
                None,
            )
            if variant:
                replacements.append((item, variant))
        if len(replacements) > 1 and replacements[0][1].options != replacements[1][1].options:
            candidates = order, replacements[:2]
            break
    assert candidates is not None
    order, pairs = candidates
    adapter.read("identity", call("find_user_id_by_email", email=db.users[order.user_id].email))
    payment = next(iter(db.users[order.user_id].payment_methods))
    with pytest.raises(ValueError, match="upstream multi-item"):
        adapter.prepare(
            call(
                "modify_pending_order_items",
                order_id=order.order_id,
                item_ids=[item.item_id for item, _ in pairs],
                new_item_ids=[variant.item_id for _, variant in pairs],
                payment_method_id=payment,
            )
        )
    assert adapter.snapshot() == db
    assert adapter.write_count == 0


def test_verification_reads_actual_committed_database(adapter):
    authenticate_emma(adapter)
    action = adapter.prepare(cancellation())
    adapter.confirm(action.fingerprint, "yes")
    result = adapter.execute(action)
    assert adapter.verify(action, result)["matches_confirmed_action"]
    assert not adapter.verify(action, {**result, "status": "pending"})["matches_confirmed_action"]
    assert adapter.write_count == 1
