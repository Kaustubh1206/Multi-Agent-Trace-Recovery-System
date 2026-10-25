"""Preserve observed facts within a request; recompute generic return finances."""

import json

from langchain_core.messages import HumanMessage
from tau2.data_model.message import AssistantMessage

from retail_benchmark.runner import run_task

from .support import (
    ScriptedAgents,
    after_write,
    plan,
    policy_review,
    proposal,
    resume,
    session,
    tool,
)
from .test_confirmation_language import customer_messages

RETURN_ITEMS = ["2872451762", "9534205511"]


def return_steps():
    return [
        plan(),
        ("identity", tool(
            "find_user_id_by_name_zip", first_name="Daiki", last_name="Johnson", zip="80273"
        )),
        ("order", tool("get_order_details", order_id="#W9502127")),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", {
            "kind": "ask", "text": "Which vacuum cleaner and which refund method?"
        }),
        plan(),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", proposal(
            "return_delivered_order_items", order_id="#W9502127",
            item_ids=RETURN_ITEMS, payment_method_id="paypal_2433177",
        )),
        *after_write(),
    ]


def test_clarification_keeps_facts_but_new_request_clears_them(adapter, dataset):
    graph, agents, config, result = session(
        adapter, dataset, return_steps(),
        "Daiki Johnson ZIP 80273, no email. Return a vacuum and air purifier in #W9502127.",
    )
    original_observations = result["observations"]
    assert original_observations
    result = resume(graph, config, "The canister vacuum and air purifier, original PayPal.")
    assert adapter.write_count == 0
    assert agents.contexts[6][0] == "supervisor"
    assert agents.contexts[6][2]["observations"] == original_observations
    assert agents.contexts[6][2]["continuing_request"] is True
    assert "return_delivered_order_items" in agents.contexts[6][2]["team_write_tools"]
    assert "1095.55" in result["__interrupt__"][0].value["message"]
    assert "1528.14" in result["__interrupt__"][0].value["message"]
    result = resume(graph, config, "Yes, please proceed with the return and refund.")
    assert adapter.write_count == 1
    assert "$1,095.55" in result["response"]
    assert "$1,528.14" in result["response"]
    assert result["verification_result"]["verified"]
    agents.steps.extend([
        plan("inquiry"),
        ("order", tool("get_order_details", order_id="#W9502127")),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", {"kind": "respond", "text": "The return is requested."}),
    ])
    graph.invoke({
        "messages": [HumanMessage(content="What is the current status?")],
        "customer_query": "What is the current status?",
        "new_request": True,
    }, config)
    new_supervisor = [context for role, _, context in agents.contexts if role == "supervisor"][-1]
    assert new_supervisor["observations"] == []
    assert new_supervisor["continuing_request"] is False
    assert adapter.write_count == 1


def test_return_amounts_come_from_purchased_prices_not_catalog_or_labels(adapter, dataset):
    from .support import authenticate_emma, call

    # Use another customer/order to ensure the arithmetic is not specific to task 47.
    authenticate_emma(adapter)
    order = adapter.snapshot().orders["#W5605613"]
    selected = [order.items[0].item_id]
    amount = order.items[0].price
    original = adapter.snapshot()
    action = adapter.prepare(call(
        "return_delivered_order_items", order_id=order.order_id,
        item_ids=selected, payment_method_id=order.payment_history[0].payment_method_id,
    ))
    assert action.financial_effect["refund_after_return"] == round(amount, 2)
    assert action.financial_effect["remaining_items_paid"] == round(
        sum(item.price for item in order.items) - amount, 2
    )
    assert adapter.snapshot() == original
    assert adapter.write_count == 0


def test_automatic_return_has_real_db_and_communication_outcomes(dataset, monkeypatch):
    import tau2.evaluator.evaluator_nl_assertions as judge

    customer_messages(monkeypatch, [
        "Daiki Johnson ZIP 80273, no email. Return a vacuum and air purifier in #W9502127.",
        "The canister vacuum and air purifier. Refund original PayPal. "
        "I need the refund quickly; can you tell me the amount and what I paid for remaining items?",
        "Yes, please proceed with the return and refund.", "###STOP###",
    ])
    expected_assertions = dataset.load_train_task("47")["evaluation_criteria"]["nl_assertions"]

    def grade(**kwargs):
        # Intercept the paid judge; native DB and substring communication checks remain real.
        prompt = kwargs["messages"][-1].content
        assert "$1,095.55" in prompt and "$1,528.14" in prompt
        return AssistantMessage(role="assistant", content=json.dumps({
            "results": [
                {"expectedOutcome": text, "metExpectation": True,
                 "reasoning": "Amounts are explicitly present in the recorded dialogue."}
                for text in expected_assertions
            ]
        }))

    monkeypatch.setattr(judge, "generate", grade)
    agents = ScriptedAgents(return_steps())
    _, report, db = run_task(dataset, "47", agents=agents)
    assert report["validated_success"]
    assert report["official_result"]["db_check"]["db_match"]
    assert all(c["met"] for c in report["official_result"]["communicate_checks"])
    assert report["committed_writes"] == 1
    assert db.orders["#W9502127"].status == "return requested"
    assert db.orders["#W9502127"].return_items == sorted(RETURN_ITEMS)
    assert not agents.steps
