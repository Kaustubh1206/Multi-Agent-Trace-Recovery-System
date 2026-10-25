"""Regressions for the natural confirmations from the failed live task 0/69 runs."""

import json

import pytest
import tau2.user.user_simulator as native_user
from tau2.data_model.message import AssistantMessage

from retail_benchmark.runner import run_task
from retail_mas.models import describe_model_error
from retail_mas.nodes import identity_agent
from retail_mas.objects import ToolCall
from retail_mas.policy import explicit_answer

from .support import (
    ScriptedAgents,
    after_write,
    authenticate_emma,
    cancellation,
    cancellation_steps,
    plan,
    policy_review,
    proposal,
    resume,
    session,
    tool,
)

CANCELLATION_REPLY = "Yes, please proceed with the cancellation and refund."
EXCHANGE_REPLY = (
    "Yes, this includes all the items I want changed. "
    "Please proceed with the exchange. Thank you!"
)


def customer_messages(monkeypatch, messages):
    replies = iter(messages)
    monkeypatch.setattr(
        native_user, "generate",
        lambda **kwargs: AssistantMessage(role="assistant", content=next(replies)),
    )


def test_live_cancellation_reply_completes_without_replanning(dataset, monkeypatch):
    customer_messages(monkeypatch, [
        "Emma Smith ZIP 10192, no email. Cancel #W2417020 because no longer needed.",
        CANCELLATION_REPLY, "###STOP###",
    ])
    agents = ScriptedAgents(cancellation_steps())
    _, report, db = run_task(dataset, "69", agents=agents)
    assert report["validated_success"]
    assert report["committed_writes"] == 1
    assert db.orders["#W2417020"].status == "cancelled"
    assert [role for role, _, _ in agents.contexts].count("supervisor") == 1
    assert not agents.steps


@pytest.mark.parametrize("task_id,include_keyboard,reply", [
    ("0", True, EXCHANGE_REPLY),
    ("0", True, "Yes, please proceed. That covers everything I want to exchange. Thanks!"),
    ("0", True, "Yes, please proceed. That includes all the items I want to exchange. Thank you!"),
    ("1", False, "Yes, proceed. That’s everything I want changed. Thank you!"),
])
def test_live_exchange_reply_completes_without_replanning(
    dataset, monkeypatch, task_id, include_keyboard, reply
):
    items = ["1151293680", "4983901480"] if include_keyboard else ["4983901480"]
    replacements = ["7706410293", "7747408585"] if include_keyboard else ["7747408585"]
    customer_messages(monkeypatch, [
        "Yusuf Rossi ZIP 19122, no email. Exchange the full-size keyboard to clicky RGB, "
        "or no backlight if unavailable, and thermostat to Google Home in #W2378156. "
        "Use my original credit card. " + (
            "I accept no backlight." if include_keyboard else "Only exchange the thermostat if RGB is unavailable."
        ),
        reply, "###STOP###",
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
            item_ids=items,
            new_item_ids=replacements,
            payment_method_id="credit_card_9513926",
        )),
        *after_write(),
    ])
    simulation, report, db = run_task(dataset, task_id, agents=agents)
    assert report["validated_success"]
    assert report["committed_writes"] == 1
    assert db.orders["#W2378156"].exchange_items == sorted(items)
    questions = [
        message.content for message in simulation.messages
        if message.role == "assistant" and message.content
        and "Proceed? Reply yes or no." in message.content
    ]
    assert len(questions) == 1
    assert [role for role, _, _ in agents.contexts].count("supervisor") == 1
    assert not agents.steps


@pytest.mark.parametrize("reply", [
    "Yes, please proceed with the cancellation and refund, but use another card.",
    "Yes, please proceed if the refund arrives today.",
    "Yes, please proceed with the exchange.",
    "Yes, cancel order #W5605613 instead.",
    "Yes, please change the reason.",
    "Yes? Please proceed with the cancellation and refund.",
    "Maybe, please go ahead.",
])
def test_changed_conditional_or_conflicting_replies_cannot_write(adapter, reply):
    authenticate_emma(adapter)
    action = adapter.prepare(cancellation())
    assert explicit_answer(reply, action.call.name) == "unclear"
    assert not adapter.confirm(action.fingerprint, reply)
    with pytest.raises(ValueError, match="approval"):
        adapter.execute(action)
    assert adapter.write_count == 0
    assert adapter.snapshot().orders["#W2417020"].status == "pending"


def test_natural_approval_still_requires_the_exact_prepared_fingerprint(adapter):
    authenticate_emma(adapter)
    action = adapter.prepare(cancellation())
    assert not adapter.confirm("wrong-fingerprint", CANCELLATION_REPLY)
    assert adapter.confirm(action.fingerprint, CANCELLATION_REPLY)
    adapter.execute(action)
    with pytest.raises(ValueError, match="already committed"):
        adapter.execute(action)
    assert adapter.write_count == 1


def test_named_confirmation_is_scoped_to_the_pending_operation():
    assert explicit_answer(CANCELLATION_REPLY, "cancel_pending_order") == "yes"
    assert explicit_answer(CANCELLATION_REPLY, "return_delivered_order_items") == "unclear"
    assert explicit_answer(EXCHANGE_REPLY, "exchange_delivered_order_items") == "yes"
    assert explicit_answer(EXCHANGE_REPLY, "cancel_pending_order") == "unclear"


def test_invalid_specialist_step_has_safe_actionable_diagnostic(adapter, dataset):
    agents = ScriptedAgents([
        ("identity", {"kind": "respond", "text": "private-customer-text"})
    ])
    with pytest.raises(ValueError) as caught:
        identity_agent({}, adapter=adapter, agents=agents, policy=dataset.policy)
    assert describe_model_error(caught.value) == "Invalid identity output kind: respond"
    assert "private-customer-text" not in describe_model_error(caught.value)
    assert describe_model_error(ValueError("private-provider-body")) == "ValueError"


def test_agent_error_reports_location_without_exception_locals(dataset, monkeypatch):
    customer_messages(monkeypatch, ["Please help me with my order."])
    agents = ScriptedAgents([
        plan(),
        ("identity", {"kind": "respond", "text": "private-customer-text"}),
    ])
    _, report, _ = run_task(dataset, "69", agents=agents)
    assert report["termination_reason"] == "agent_error"
    assert report["run_error"] == "Invalid identity output kind: respond"
    assert report["run_error_location"]["module"] == "retail_mas.nodes"
    assert report["run_error_location"]["function"] == "agent_step"
    assert isinstance(report["run_error_location"]["line"], int)
    assert "private-customer-text" not in json.dumps(report)


@pytest.mark.parametrize("reply,action_name", [
    (
        "Yes, please proceed. That includes all the items I want to return. Thanks!",
        "return_delivered_order_items",
    ),
    (
        "Yes, please proceed. That covers everything I want to change.",
        "modify_pending_order_items",
    ),
    (
        "Yes, that includes all items I want to exchange. Please proceed.",
        "exchange_delivered_order_items",
    ),
])
def test_item_acknowledgements_are_scoped_to_the_pending_operation(reply, action_name):
    assert explicit_answer(reply, action_name) == "yes"
    assert explicit_answer(reply, "cancel_pending_order") == "unclear"


@pytest.mark.parametrize("reply", [
    "Yes, please proceed. That covers everything I want to return. Thanks!",
    "Yes, please proceed. That covers everything I want to exchange except the keyboard.",
    "Yes, please proceed. That covers everything I want to exchange if shipping is free.",
    "Yes, please proceed. That covers everything I want to exchange. Use another card.",
    "Yes, please proceed. That covers everything I want to exchange in order #W2417020.",
    "Yes? Please proceed. That covers everything I want to exchange.",
    "That covers everything I want to exchange. Thanks!",
    "Yes, please proceed. That includes all the items I want to exchange. ###STOP###",
])
def test_item_acknowledgement_cannot_hide_changed_or_missing_approval(adapter, reply):
    adapter.read("identity", ToolCall(
        name="find_user_id_by_name_zip",
        arguments={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
    ))
    action = adapter.prepare(ToolCall(
        name="exchange_delivered_order_items",
        arguments={
            "order_id": "#W2378156",
            "item_ids": ["1151293680", "4983901480"],
            "new_item_ids": ["7706410293", "7747408585"],
            "payment_method_id": "credit_card_9513926",
        },
    ))
    before = adapter.snapshot()
    assert explicit_answer(reply, action.call.name) == "unclear"
    assert not adapter.confirm(action.fingerprint, reply)
    with pytest.raises(ValueError, match="approval"):
        adapter.execute(action)
    assert adapter.write_count == 0
    assert adapter.snapshot() == before


def test_simulator_stop_with_approval_does_not_execute_a_pending_action(dataset, monkeypatch):
    customer_messages(monkeypatch, [
        "Emma Smith ZIP 10192, no email. Cancel #W2417020 because no longer needed.",
        CANCELLATION_REPLY + "\n\n###STOP###",
    ])
    agents = ScriptedAgents(cancellation_steps(confirmed=False))
    _, report, db = run_task(dataset, "69", agents=agents)
    assert report["termination_reason"] == "user_stop"
    assert report["evaluation_status"] == "complete"
    assert report["committed_writes"] == 0
    assert report["initial_database_hash"] == report["final_database_hash"]
    assert db.orders["#W2417020"].status == "pending"
    assert not agents.steps


INFORMAL_EXCHANGE_CONFIRMATION = (
    "I have the requested exchange details: order #W2378156, exchange only thermostat "
    "item 4983901480 for available black Google Assistant thermostat item 7747408585, "
    "using your Mastercard ending 2478 for any price difference. The keyboard will "
    "remain unchanged, and the replacement is listed as Google Assistant compatible "
    "(not specifically Google Home). Please confirm that this is the exact replacement "
    "you want and that the thermostat is the only item you want exchanged—all items "
    "to exchange have been listed."
)


@pytest.mark.parametrize("origin,kind", [
    ("supervisor", "question"),
    ("order", "ask"),
    ("decision", "ask"),
    ("decision", "respond"),
])
def test_informal_approval_is_corrected_before_reaching_customer(
    adapter, dataset, record_pilot, origin, kind
):
    steps = [
        plan(),
        ("identity", tool(
            "find_user_id_by_name_zip", first_name="Yusuf", last_name="Rossi", zip="19122"
        )),
        ("order", tool("get_order_details", order_id="#W2378156")),
        ("order", tool("get_product_details", product_id="1656367028")),
        ("order", tool("get_product_details", product_id="4896585277")),
        ("order", tool("get_user_details", user_id="yusuf_rossi_9620")),
    ]
    bad_step = {"kind": kind, "text": INFORMAL_EXCHANGE_CONFIRMATION}
    if origin == "supervisor":
        steps[0][1]["question"] = INFORMAL_EXCHANGE_CONFIRMATION
    if origin == "order":
        steps.append(("order", bad_step))
    steps.extend([("order", {"kind": "complete"}), policy_review()])
    if origin == "decision":
        steps.append(("decision", bad_step))
    steps.extend([
        ("decision", proposal(
            "exchange_delivered_order_items", order_id="#W2378156",
            item_ids=["4983901480"], new_item_ids=["7747408585"],
            payment_method_id="credit_card_9513926",
        )),
        *after_write(),
    ])
    before = adapter.snapshot()
    graph, agents, config, result = session(
        adapter, dataset, steps,
        "Yusuf Rossi ZIP 19122, no email. Since full-size clicky RGB is unavailable, "
        "keep my keyboard. Yes, exchange only the thermostat in #W2378156 for the "
        "black Google Assistant model, using my original Mastercard.",
        "1",
    )
    pause = result["__interrupt__"][0].value
    assert pause["kind"] == "confirmation"
    assert pause["message"].startswith("Proposed exchange_delivered_order_items:")
    assert INFORMAL_EXCHANGE_CONFIRMATION not in pause["message"]
    assert result["pending_action"].call.arguments["item_ids"] == ["4983901480"]
    assert adapter.write_count == 0
    assert adapter.snapshot() == before
    if origin != "supervisor":
        corrections = [
            context["validation_error"] for role, _, context in agents.contexts
            if role == origin and context.get("validation_error")
        ]
        assert len(corrections) == 1
        assert "Do not ask for action approval in prose" in corrections[0]
    final = resume(graph, config, "Yes, please proceed with the exchange.")
    assert adapter.write_count == 1
    assert final["verification_result"]["verified"]
    assert adapter.snapshot().orders["#W2378156"].exchange_items == ["4983901480"]
    assert not any(
        message.content == INFORMAL_EXCHANGE_CONFIRMATION for message in final["messages"]
    )
    assert not agents.steps
    record_pilot("1", adapter)


def test_repeated_unprepared_approval_requests_fail_closed_with_a_bounded_retry(
    dataset, monkeypatch
):
    customer_messages(monkeypatch, [
        "Emma Smith ZIP 10192, no email. Cancel #W2417020 because no longer needed.",
    ])
    approval_question = {"kind": "ask", "text": "Please confirm the cancellation."}
    agents = ScriptedAgents([
        *cancellation_steps(confirmed=False)[:-1],
        ("decision", approval_question),
        ("decision", approval_question),
    ])
    _, report, db = run_task(dataset, "69", agents=agents)
    assert report["termination_reason"] == "agent_error"
    assert report["run_error"] == (
        "decision repeated an unprepared approval request; no action taken"
    )
    assert report["committed_writes"] == 0
    assert report["initial_database_hash"] == report["final_database_hash"]
    assert db.orders["#W2417020"].status == "pending"
    assert [role for role, _, _ in agents.contexts].count("decision") == 2
    assert not agents.steps
