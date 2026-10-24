import pytest
from langchain_core.messages import HumanMessage

from retail_mas.nodes import TRANSFER_MESSAGE
from retail_mas.tools import RetailAdapter

from .support import cancellation_steps, plan, proposal, resume, reviews, session, tool


def test_read_only_real_order_is_unchanged(adapter, dataset):
    original = adapter.snapshot()
    steps = [
        plan("inquiry"),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Emma", last_name="Smith", zip="10192"),
        ),
        ("order", tool("get_order_details", order_id="#W2417020")),
        ("order", {"kind": "complete"}),
        ("decision", {"kind": "respond", "text": "Your laptop order #W2417020 is pending."}),
    ]
    _, agents, _, result = session(
        adapter,
        dataset,
        steps,
        "I'm Emma Smith, ZIP 10192; I cannot remember email. What is the status of #W2417020?",
    )
    assert "pending" in result["response"]
    assert not agents.steps
    assert adapter.snapshot() == original
    assert adapter.write_count == 0


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_cancellation_real_interrupt(adapter, dataset, answer):
    graph, agents, config, result = session(
        adapter,
        dataset,
        cancellation_steps(),
        "I'm Emma Smith ZIP 10192, no email. Cancel #W2417020: no longer needed.",
        "69",
    )
    pause = result["__interrupt__"][0].value
    assert pause["kind"] == "confirmation"
    assert "#W2417020" in pause["message"]
    assert "2674.4" in pause["message"]
    assert adapter.snapshot().orders["#W2417020"].status == "pending"
    assert adapter.write_count == 0
    final = resume(graph, config, answer)
    assert not agents.steps
    assert adapter.write_count == int(answer == "yes")
    assert adapter.snapshot().orders["#W2417020"].status == (
        "cancelled" if answer == "yes" else "pending"
    )
    assert not final.get("__interrupt__")


def test_ambiguous_confirmation_goes_back_to_customer_without_write(adapter, dataset):
    steps = [
        *cancellation_steps(),
        plan(),
        ("order", {"kind": "ask", "text": "Please clarify the requested action."}),
    ]
    graph, _, config, result = session(
        adapter, dataset, steps, "Emma Smith, ZIP 10192; cancel #W2417020 because no longer needed."
    )
    assert result["__interrupt__"][0].value["kind"] == "confirmation"
    result = resume(graph, config, "perhaps, I am not sure")
    assert result["__interrupt__"][0].value["kind"] == "customer"
    assert result["pending_action"] is None
    assert adapter.write_count == 0


def test_task_69_return_falls_back_to_cancel(adapter, dataset, record_pilot):
    steps = [
        plan(),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Emma", last_name="Smith", zip="10192"),
        ),
        ("order", tool("get_user_details", user_id="emma_smith_8564")),
        ("order", tool("get_order_details", order_id="#W2417020")),
        ("order", {"kind": "complete"}),
        (
            "decision",
            {
                "kind": "ask",
                "text": "The laptop order is pending, so cannot be returned. Cancel instead? Please give the cancellation reason.",
            },
        ),
        plan(),
        ("order", tool("get_order_details", order_id="#W2417020")),
        ("order", {"kind": "complete"}),
        (
            "decision",
            proposal("cancel_pending_order", order_id="#W2417020", reason="no longer needed"),
        ),
        *reviews(),
    ]
    graph, agents, config, result = session(
        adapter,
        dataset,
        steps,
        "I am Emma Smith, ZIP 10192; no email. I want to return my laptop. If it cannot be returned, can it be cancelled?",
        "69",
    )
    result = resume(graph, config, "Cancel it because it is no longer needed.")
    assert result["__interrupt__"][0].value["kind"] == "confirmation"
    result = resume(graph, config, "yes")
    assert adapter.write_count == 1
    assert result["execution_result"]["status"] == "cancelled"
    assert not agents.steps
    record_pilot("69", adapter)


@pytest.mark.parametrize("task_id,include_keyboard", [("0", True), ("1", False)])
def test_exchange_preferences_and_single_multi_item_operation(
    adapter, dataset, task_id, include_keyboard, record_pilot
):
    items = ["1151293680", "4983901480"] if include_keyboard else ["4983901480"]
    new_items = ["7706410293", "7747408585"] if include_keyboard else ["7747408585"]
    request = "If there is no full size clicky RGB keyboard, " + (
        "I accept no backlight." if include_keyboard else "only exchange the thermostat."
    )
    steps = [
        plan(),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Yusuf", last_name="Rossi", zip="19122"),
        ),
        ("order", tool("get_order_details", order_id="#W2378156")),
        ("order", tool("get_product_details", product_id="1656367028")),
        ("order", tool("get_product_details", product_id="4896585277")),
        ("order", tool("get_user_details", user_id="yusuf_rossi_9620")),
        ("order", {"kind": "complete"}),
        (
            "decision",
            proposal(
                "exchange_delivered_order_items",
                order_id="#W2378156",
                item_ids=items,
                new_item_ids=new_items,
                payment_method_id="credit_card_9513926",
            ),
        ),
        *reviews(),
    ]
    original = adapter.snapshot()
    graph, agents, config, result = session(
        adapter,
        dataset,
        steps,
        "Yusuf Rossi, ZIP 19122, no email. Exchange the keyboard to clicky and thermostat to Google Home in #W2378156. "
        + request
        + " Use my Mastercard for the price difference.",
        task_id,
    )
    summary = result["__interrupt__"][0].value["message"]
    assert "ALL items" in summary
    assert adapter.write_count == 0
    result = resume(graph, config, "yes")
    assert adapter.write_count == 1
    assert result["execution_result"]["exchange_items"] == sorted(items)
    assert adapter.snapshot().products == original.products
    assert adapter.snapshot().users == original.users
    assert not agents.steps
    record_pilot(task_id, adapter)


def test_task_47_wrong_reference_requires_customer_correction(adapter, dataset, record_pilot):
    steps = [
        plan(),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Daiki", last_name="Johnson", zip="80273"),
        ),
        ("order", tool("get_order_details", order_id="#9502126")),
        ("order", {"kind": "ask", "text": "Please double-check the order reference."}),
        plan(),
        ("order", tool("get_order_details", order_id="#9502127")),
        (
            "order",
            {"kind": "ask", "text": "That reference is unavailable too. Please check the prefix."},
        ),
        plan(),
        ("order", tool("get_order_details", order_id="#W9502127")),
        ("order", tool("calculate", expression="622.12 + 473.43")),
        ("order", {"kind": "complete"}),
        (
            "decision",
            proposal(
                "return_delivered_order_items",
                order_id="#W9502127",
                item_ids=["2872451762", "9534205511"],
                payment_method_id="paypal_2433177",
            ),
        ),
        *reviews(),
        plan("inquiry"),
        ("order", tool("get_order_details", order_id="#W9502127")),
        ("order", tool("calculate", expression="2623.69 - 1095.55")),
        ("order", {"kind": "complete"}),
        (
            "decision",
            {
                "kind": "respond",
                "text": "The requested refund is $1,095.55. You paid $1,528.14 for remaining items. I cannot promise a refund within 3 days.",
            },
        ),
    ]
    graph, agents, config, result = session(
        adapter,
        dataset,
        steps,
        "Daiki Johnson, ZIP 80273; cannot remember email. Return air purifier and canister vacuum, order 9502126, original PayPal.",
        "47",
    )
    result = resume(graph, config, "I made a mistake: 9502127")
    assert adapter.write_count == 0
    result = resume(graph, config, "I forgot the W: #W9502127")
    assert result["pending_action"].call.arguments["order_id"] == "#W9502127"
    assert adapter.write_count == 0
    result = resume(graph, config, "yes")
    assert adapter.write_count == 1
    query = "How much is the refund, and how much did I pay for the remaining items? Can it arrive within 3 days?"
    result = graph.invoke(
        {"customer_query": query, "messages": [HumanMessage(content=query)]}, config
    )
    assert "1,528.14" in result["response"]
    assert adapter.write_count == 1
    assert not agents.steps
    record_pilot("47", adapter)


def test_task_84_changed_item_and_payment_needs_fresh_consent(adapter, dataset, record_pilot):
    def order_reads():
        return [
            ("order", tool("get_order_details", order_id="#W3069600")),
            ("order", tool("get_order_details", order_id="#W9571698")),
            ("order", tool("get_user_details", user_id="chen_silva_7485")),
            ("order", {"kind": "complete"}),
        ]

    steps = [
        plan(),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Chen", last_name="Silva", zip="46281"),
        ),
        *order_reads(),
        (
            "decision",
            proposal(
                "return_delivered_order_items",
                order_id="#W3069600",
                item_ids=["8551474201"],
                payment_method_id="credit_card_1565124",
            ),
        ),
        *reviews(),
        plan(),
        *order_reads(),
        (
            "decision",
            proposal(
                "return_delivered_order_items",
                order_id="#W9571698",
                item_ids=["6065192424"],
                payment_method_id="gift_card_7250692",
            ),
        ),
        *reviews(),
    ]
    graph, agents, config, result = session(
        adapter,
        dataset,
        steps,
        "Chen Silva ZIP 46281, no email. I received two tablets. Return the cheaper one to my credit card.",
        "84",
    )
    old_fingerprint = result["pending_action"].fingerprint
    result = resume(
        graph, config, "Actually return the more expensive tablet and refund my existing gift card."
    )
    assert adapter.write_count == 0
    assert result["pending_action"].fingerprint != old_fingerprint
    assert result["__interrupt__"][0].value["kind"] == "confirmation"
    result = resume(graph, config, "yes")
    db = adapter.snapshot()
    assert db.orders["#W3069600"].status == "delivered"
    assert db.orders["#W9571698"].return_items == ["6065192424"]
    assert adapter.write_count == 1
    assert not agents.steps
    record_pilot("84", adapter)


def test_task_50_transfer_uses_real_tool(adapter, dataset, monkeypatch, record_pilot):
    calls = []
    transfer = adapter.transfer

    def record(summary):
        calls.append(summary)
        return transfer(summary)

    monkeypatch.setattr(adapter, "transfer", record)
    original = adapter.snapshot()
    steps = [
        plan("unsupported"),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Chen", last_name="Smith", zip="32278"),
        ),
        ("order", {"kind": "complete"}),
        (
            "decision",
            {
                "kind": "transfer",
                "text": "Customer insists on reversing a cancellation and urgent delivery; no supported tool can do this.",
            },
        ),
    ]
    _, agents, _, result = session(
        adapter,
        dataset,
        steps,
        "Chen Smith ZIP 32278, no email. Undo my cancellation and deliver it urgently.",
        "50",
    )
    assert result["response"] == TRANSFER_MESSAGE
    assert len(calls) == 1
    assert adapter.snapshot() == original
    assert not agents.steps
    record_pilot("50", adapter)


def test_sequential_tasks_do_not_share_database_or_consent(dataset):
    first = RetailAdapter(dataset.fresh_db())
    graph, _, config, _ = session(
        first,
        dataset,
        cancellation_steps(),
        "Emma Smith ZIP 10192, no email. Cancel #W2417020, no longer needed.",
        "69",
    )
    resume(graph, config, "yes")
    second = RetailAdapter(dataset.fresh_db())
    assert second.user_id is None
    graph, _, config, result = session(
        second,
        dataset,
        cancellation_steps(),
        "Emma Smith ZIP 10192, no email. Cancel #W2417020, no longer needed.",
        "69",
    )
    assert result["__interrupt__"]
    assert second.write_count == 0
    assert second.snapshot().orders["#W2417020"].status == "pending"
    resume(graph, config, "no")
    assert second.write_count == 0


def test_step_budget_prevents_unbounded_model_loops(adapter, dataset):
    from retail_mas.nodes import MAX_STEPS

    steps = [
        plan("inquiry"),
        (
            "identity",
            tool("find_user_id_by_name_zip", first_name="Emma", last_name="Smith", zip="10192"),
        ),
        *[("order", tool("get_order_details", order_id="#W2417020")) for _ in range(MAX_STEPS - 2)],
    ]
    _, agents, _, result = session(
        adapter, dataset, steps, "Emma Smith ZIP 10192, no email. Order status?"
    )
    assert "budget" in result["response"]
    assert not agents.steps
    assert adapter.write_count == 0
