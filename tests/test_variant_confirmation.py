"""Real tool/evaluator regressions for a clarification guard and variant mismatch."""

import pytest

from retail_benchmark.runner import run_task
from retail_mas.objects import ToolCall
from retail_mas.policy import confirmation_summary

from .support import (
    ScriptedAgents,
    cancellation,
    plan,
    policy_review,
    proposal,
    tool,
)
from .test_confirmation_language import customer_messages

APPROVAL = (
    "Yes, please proceed with the exchange as described. "
    "This covers everything I wanted to change. Thank you!"
)


@pytest.mark.parametrize("question", [
    "The laptop order is pending. Would you like me to cancel it instead? "
    "What is the cancellation reason?",
    "The order is pending. Please confirm the cancellation reason: "
    "no longer needed or ordered by mistake.",
    "Return isn't available for the pending order. Please confirm whether "
    "you want cancellation instead.",
])
def test_ineligible_return_can_clarify_a_cancellation_without_approval_guard_error(
    dataset, monkeypatch, question
):
    customer_messages(monkeypatch, [
        "Emma Smith ZIP 10192, no email. I'd like to return my laptop.",
        "Cancel the laptop order instead because it is no longer needed.",
        "Yes, please proceed with the cancellation and refund.",
        "###STOP###",
    ])
    agents = ScriptedAgents([
        plan(),
        ("identity", tool(
            "find_user_id_by_name_zip", first_name="Emma", last_name="Smith", zip="10192"
        )),
        ("order", tool("get_order_details", order_id="#W2417020")),
        ("order", {"kind": "complete"}),
        ("policy", {"allowed": False, "reason": "Pending orders cannot be returned."}),
        ("decision", {"kind": "ask", "text": question}),
        plan(),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", {"kind": "propose", "tool": cancellation().model_dump()}),
    ])
    simulation, report, db = run_task(dataset, "69", agents=agents)
    assert report["termination_reason"] == "user_stop"
    assert report["run_error"] is None
    assert report["validated_success"]
    assert report["committed_writes"] == 1
    assert db.orders["#W2417020"].status == "cancelled"
    assert sum(role == "decision" for role, _, _ in agents.contexts) == 2
    assert any(m.role == "assistant" and m.content == question for m in simulation.messages)
    assert not agents.steps


@pytest.mark.parametrize("keyboard_id,expected_reward,backlight", [
    ("7706410293", 1.0, "none"),
    ("6342039236", 0.0, "white"),
])
def test_confirmed_exchange_keeps_native_wrong_variant_failure(
    dataset, monkeypatch, keyboard_id, expected_reward, backlight
):
    # The evaluator's reference is not used to choose a variant at runtime.
    customer_messages(monkeypatch, [
        "Yusuf Rossi ZIP 19122, no email. Exchange the keyboard and thermostat "
        "in #W2378156 using my original Mastercard.",
        APPROVAL, "###STOP###",
    ])
    agents = ScriptedAgents([
        plan(),
        ("identity", tool(
            "find_user_id_by_name_zip", first_name="Yusuf", last_name="Rossi", zip="19122"
        )),
        ("order", tool("get_order_details", order_id="#W2378156")),
        ("order", tool("get_product_details", product_id="1656367028")),
        ("order", tool("get_product_details", product_id="4896585277")),
        ("order", tool("get_user_details", user_id="yusuf_rossi_9620")),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", proposal(
            "exchange_delivered_order_items", order_id="#W2378156",
            item_ids=["1151293680", "4983901480"],
            new_item_ids=[keyboard_id, "7747408585"],
            payment_method_id="credit_card_9513926",
        )),
    ])
    simulation, report, _ = run_task(dataset, "0", agents=agents)
    assert report["native_reward"] == expected_reward
    assert report["validated_success"] == (expected_reward == 1)
    assert report["committed_writes"] == 1
    assert report["supplementary_checks"]["exact_confirmation_before_writes"]
    questions = [
        m.content for m in simulation.messages
        if m.role == "assistant" and m.content and "Proceed? Reply yes or no." in m.content
    ]
    assert len(questions) == 1
    assert f'"backlight": "{backlight}"' in questions[0]
    assert not agents.steps


def test_exchange_summary_options_are_observed_copies_bound_to_approval(adapter):
    adapter.read("identity", ToolCall(
        name="find_user_id_by_name_zip",
        arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
    ))
    initial = adapter.snapshot()
    action = adapter.prepare(ToolCall(
        name="exchange_delivered_order_items",
        arguments={
            "order_id": "#W2378156", "item_ids": ["1151293680"],
            "new_item_ids": ["6342039236"], "payment_method_id": "credit_card_9513926",
        },
    ))
    change = action.financial_effect["item_changes"][0]
    assert change["from_options"] == {
        "switch type": "linear", "backlight": "RGB", "size": "full size",
    }
    assert change["to_options"] == {
        "switch type": "clicky", "backlight": "white", "size": "full size",
    }
    assert change["replacement_price"] == 244.91
    assert '"backlight": "white"' in confirmation_summary(action)
    original_fingerprint = action.fingerprint
    change["to_options"]["backlight"] = "RGB"
    assert action.fingerprint != original_fingerprint
    assert adapter.snapshot() == initial
    assert not adapter.confirm(action.fingerprint, APPROVAL)
    with pytest.raises(ValueError, match="approval"):
        adapter.execute(action)
    assert adapter.write_count == 0
