"""Official simulator/evaluator code runs offline; only model responses are scripted."""

import json
from collections import deque
from copy import deepcopy

import pytest
import tau2.evaluator.evaluator_nl_assertions as native_judge
import tau2.user.user_simulator as native_user
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun

from retail_benchmark.runner import run_task

from .support import (
    ScriptedAgents,
    after_write,
    cancellation_steps,
    plan,
    policy_review,
    proposal,
    tool,
)

REQUEST = "Emma Smith, ZIP 10192, no email. Cancel #W2417020 because it is no longer needed."


@pytest.fixture
def user_responses(monkeypatch):
    calls = []

    def install(replies):
        pending = deque(replies)

        def generate(**kwargs):
            calls.append(deepcopy(kwargs))
            content = pending.popleft()
            if isinstance(content, Exception):
                raise content
            return AssistantMessage(role="assistant", content=content)

        monkeypatch.setattr(native_user, "generate", generate)
        return pending, calls

    return install


def test_automatic_confirmation_uses_official_simulator_and_native_all(dataset, user_responses):
    pending, calls = user_responses([REQUEST, "yes", "###STOP###"])
    agents = ScriptedAgents(cancellation_steps())
    simulation, report, final_db = run_task(dataset, "69", agents=agents)
    assert not pending and not agents.steps
    assert final_db.orders["#W2417020"].status == "cancelled"
    assert report["native_reward"] == 1.0
    assert report["validated_success"]
    assert report["committed_writes"] == 1
    assert report["evaluation_status"] == "complete"
    assert all(report["supplementary_checks"].values())
    assert report["execution_mode"] == "offline_scripted_agents"
    assert report["changed_records"]["orders"] == ["#W2417020"]
    assert report["changed_records"]["users"] == ["emma_smith_8564"]
    assert report["changed_records"]["products"] == []
    assert simulation.termination_reason.value == "user_stop"
    assert calls[0]["model"] == "openai/gpt-4.1-2025-04-14"
    assert calls[0]["seed"] == 300
    assert "# User Simulation Guidelines" in calls[0]["messages"][0].content
    assert "Reason for call:" in calls[0]["messages"][0].content
    # The official simulator flips roles and receives the actual confirmation question.
    assert calls[1]["messages"][-1].role == "user"
    assert "Proceed? Reply yes or no." in calls[1]["messages"][-1].content
    tools = [m for m in simulation.messages if m.role == "assistant" and m.is_tool_call()]
    assert all(len(message.tool_calls) == 1 and not message.content for message in tools)
    assert sum(m.tool_calls[0].name == "cancel_pending_order" for m in tools) == 1
    assert SimulationRun.model_validate_json(simulation.model_dump_json()) == simulation


def test_no_confirmation_and_sequential_trials_are_isolated(dataset, user_responses):
    user_responses([REQUEST, "yes", "###STOP###"])
    first, _, first_db = run_task(dataset, "69", agents=ScriptedAgents(cancellation_steps()))
    user_responses([REQUEST, "no", "###STOP###"])
    second, report, second_db = run_task(
        dataset, "69", trial=2, seed=301, agents=ScriptedAgents(cancellation_steps(False))
    )
    assert first.id != second.id
    assert first_db.orders["#W2417020"].status == "cancelled"
    assert second_db.orders["#W2417020"].status == "pending"
    assert report["committed_writes"] == 0
    assert report["native_reward"] == 0.0
    assert not report["validated_success"]
    assert report["initial_database_hash"] == report["final_database_hash"]


def test_ambiguous_approval_and_turn_budget_do_not_write(dataset, user_responses):
    user_responses([REQUEST, "perhaps"])
    agents = ScriptedAgents([
        *cancellation_steps(False),
        plan(),
        ("order", {"kind": "ask", "text": "Please clarify what you want to do."}),
    ])
    _, report, db = run_task(dataset, "69", agents=agents, max_turns=2)
    assert db.orders["#W2417020"].status == "pending"
    assert report["termination_reason"] == "max_steps"
    assert report["committed_writes"] == 0
    assert report["evaluation_status"] == "incomplete_run"
    assert report["native_reward"] == 0.0


@pytest.mark.parametrize("failure_stage", ["user", "agent"])
def test_model_failures_produce_incomplete_reports(dataset, user_responses, failure_stage):
    user_responses(
        [ConnectionError("private-provider-body")] if failure_stage == "user" else [REQUEST]
    )
    _, report, _ = run_task(dataset, "69", agents=ScriptedAgents([]))
    assert report["termination_reason"] == f"{failure_stage}_error"
    assert report["evaluation_status"] == "incomplete_run"
    assert not report["validated_success"]
    assert report["committed_writes"] == 0
    assert "private-provider-body" not in json.dumps(report)


@pytest.mark.parametrize("transfer", [True, False])
def test_required_handoff_checked_beyond_native_db_reward(dataset, user_responses, transfer):
    user_responses([
        "Chen Smith ZIP 32278, no email. Undo my cancellation and deliver it urgently.",
        "###TRANSFER###" if transfer else "###STOP###",
    ])
    agents = ScriptedAgents([
        plan("unsupported"),
        ("identity", tool(
            "find_user_id_by_name_zip", first_name="Chen", last_name="Smith", zip="32278"
        )),
        policy_review(),
        ("decision", {
            "kind": "transfer" if transfer else "respond", "text": "Unsupported request."
        }),
    ])
    _, report, _ = run_task(dataset, "50", agents=agents)
    # Native DB equality alone passes this task; the separate handoff check must still apply.
    assert report["native_reward"] == 1.0
    assert report["human_transfers"] == int(transfer)
    assert report["supplementary_checks"]["required_human_handoff"] == transfer
    assert report["validated_success"] == transfer
    assert report["committed_writes"] == 0


def test_simulator_private_instructions_and_criteria_never_enter_mas(dataset, user_responses):
    _, calls = user_responses([REQUEST, "yes", "###STOP###"])
    agents = ScriptedAgents(cancellation_steps())
    simulation, _, _ = run_task(dataset, "69", agents=agents)
    private = dataset.load_train_task("69")["user_scenario"]["instructions"]["task_instructions"]
    assert private in calls[0]["messages"][0].content
    for serialized in (json.dumps(agents.contexts), simulation.model_dump_json()):
        for forbidden in (
            private, "evaluation_criteria", "known_info", "unknown_info", "user_scenario"
        ):
            assert forbidden not in serialized


def inquiry_steps():
    return [
        plan("inquiry"),
        ("identity", tool(
            "find_user_id_by_name_zip", first_name="Daiki", last_name="Johnson", zip="80273"
        )),
        ("order", tool("get_order_details", order_id="#W9502127")),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", {
            "kind": "respond", "text": "Refund $1,095.55; remaining items $1,528.14."
        }),
    ]


@pytest.mark.parametrize("judge_mode", ["valid", "missing_assertions", "error"])
def test_native_communication_and_nl_judge_reports(
    dataset, user_responses, monkeypatch, judge_mode
):
    user_responses([
        "Daiki Johnson ZIP 80273, no email. Tell me about #W9502127.", "###STOP###"
    ])
    assertions = dataset.load_train_task("47")["evaluation_criteria"]["nl_assertions"]
    judge_calls = []

    def generate(**kwargs):
        judge_calls.append(kwargs)
        if judge_mode == "error":
            raise ConnectionError("private-judge-body")
        results = [
            {"expectedOutcome": value, "metExpectation": True, "reasoning": "Communicated"}
            for value in assertions
        ] if judge_mode == "valid" else []
        return AssistantMessage(role="assistant", content=json.dumps({"results": results}))

    monkeypatch.setattr(native_judge, "generate", generate)
    _, report, _ = run_task(dataset, "47", agents=ScriptedAgents(inquiry_steps()))
    assert len(judge_calls) == 1
    assert report["official_env_result"]["db_check"]["db_match"] is False
    assert not report["validated_success"]  # Deliberately no return was executed.
    if judge_mode == "error":
        assert report["official_result"] is None
        assert report["evaluation_status"] == "error"
        assert report["native_reward"] is None
        assert "private-judge-body" not in json.dumps(report)
    else:
        assert all(c["met"] for c in report["official_result"]["communicate_checks"])
        assert report["supplementary_checks"]["nl_judge_coverage"] == (judge_mode == "valid")
        assert report["evaluation_status"] == (
            "complete" if judge_mode == "valid" else "invalid_judge_result"
        )


def test_held_out_tasks_rejected_before_any_model_call(dataset, user_responses):
    pending, calls = user_responses([])
    with pytest.raises(ValueError, match="train"):
        run_task(dataset, "5", agents=ScriptedAgents([]))
    assert not calls and not pending


@pytest.mark.parametrize("task_id,exchange_keyboard", [("0", True), ("1", False)])
def test_exchange_preferences_survive_automatic_customer_turns(
    dataset, user_responses, task_id, exchange_keyboard
):
    fallback = (
        "I accept no backlight."
        if exchange_keyboard else "Leave the keyboard unchanged if RGB isn't available."
    )
    user_responses([
        "Yusuf Rossi ZIP 19122, no email. Exchange my keyboard to full size clicky RGB and "
        "thermostat to Google Home in #W2378156. " + fallback + " Use the original credit card.",
        "yes", "###STOP###",
    ])
    items = ["1151293680", "4983901480"] if exchange_keyboard else ["4983901480"]
    replacements = ["7706410293", "7747408585"] if exchange_keyboard else ["7747408585"]
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
            "exchange_delivered_order_items", order_id="#W2378156", item_ids=items,
            new_item_ids=replacements, payment_method_id="credit_card_9513926"
        )),
        *after_write(),
    ])
    _, report, db = run_task(dataset, task_id, agents=agents)
    assert not agents.steps
    assert report["validated_success"]
    assert report["committed_writes"] == 1
    assert db.orders["#W2378156"].status == "exchange requested"
    assert db.orders["#W2378156"].exchange_items == sorted(items)
