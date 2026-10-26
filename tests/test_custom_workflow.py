"""Cost controls, multi-order consent and the fixed development/baseline split."""

import json
import os
import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage
from tau2.data_model.message import AssistantMessage

from retail_benchmark.runner import run_task
from retail_mas.data_loader import ROOT
from retail_mas.models import AgentModels, ModelOutputError
from retail_mas.objects import PolicyDecision
from retail_mas.prompts import system_prompt

from .support import ScriptedAgents, plan, policy_review, proposal, resume, session, tool
from .test_confirmation_language import customer_messages


class UsageModel:
    def __init__(self, content='{"allowed": true, "reason": "eligible"}'):
        self.calls = 0
        self.content = content

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.content, usage_metadata={
            "input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
            "input_token_details": {"cache_read": 4},
        })


def test_usage_and_budget_include_retries_without_extra_paid_requests():
    model = UsageModel()
    agents = AgentModels(model, output_method="json", max_calls=2)
    for role in ("policy", "decision"):
        agents.invoke(role, PolicyDecision, "test", {})
    with pytest.raises(ModelOutputError, match="budget reached"):
        agents.invoke("policy", PolicyDecision, "test", {})
    assert model.calls == agents.usage["calls"] == 2
    assert agents.usage["input_tokens"] == 20
    assert agents.usage["output_tokens"] == 6
    assert agents.usage["cached_input_tokens"] == 8
    assert agents.usage["responses_with_usage"] == 2
    assert agents.usage["per_role"]["policy"]["calls"] == 1
    assert agents.usage["per_role"]["decision"]["calls"] == 1
    invalid = UsageModel("not JSON")
    with pytest.raises(ModelOutputError, match="budget reached"):
        AgentModels(invalid, output_method="json", max_calls=1).invoke(
            "policy", PolicyDecision, "test", {}
        )
    assert invalid.calls == 1


def test_full_policy_is_supplied_only_to_the_policy_llm():
    policy = "PINNED_POLICY_SENTINEL_" * 100
    for role in ("supervisor", "identity", "order", "decision"):
        assert policy not in system_prompt(role, policy)
    assert policy in system_prompt("policy", policy)


def queued_cancellations():
    step = proposal(
        "cancel_pending_order", order_id="#W5056519", reason="ordered by mistake"
    )
    step["following_actions"] = [{
        "name": "cancel_pending_order",
        "arguments": {"order_id": "#W5995614", "reason": "ordered by mistake"},
    }]
    return [
        plan(),
        ("identity", tool(
            "find_user_id_by_name_zip", first_name="Yara", last_name="Muller", zip="85041"
        )),
        ("order", tool("get_order_details", order_id="#W5056519")),
        ("order", tool("get_order_details", order_id="#W5995614")),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", step),
    ]


@pytest.mark.parametrize("second_reply", ["yes", "no"])
def test_each_queued_write_requires_its_own_confirmation_and_verification(
    dataset, monkeypatch, second_reply
):
    import tau2.evaluator.evaluator_nl_assertions as judge

    customer_messages(monkeypatch, [
        "Yara Muller ZIP 85041, no email. Cancel all pending orders: ordered by mistake.",
        "yes", second_reply, "###STOP###",
    ])
    assertions = dataset.load_train_task("113")["evaluation_criteria"]["nl_assertions"]
    monkeypatch.setattr(judge, "generate", lambda **kwargs: AssistantMessage(
        role="assistant", content=json.dumps({"results": [
            {"expectedOutcome": text, "metExpectation": second_reply == "yes",
             "reasoning": "Offline judge stub; DB and consent checks remain native."}
            for text in assertions
        ]})
    ))
    agents = ScriptedAgents(queued_cancellations())
    simulation, report, db = run_task(dataset, "113", agents=agents)
    expected = 2 if second_reply == "yes" else 1
    assert report["committed_writes"] == expected
    assert report["supplementary_checks"]["writes_verified"]
    assert report["supplementary_checks"]["exact_confirmation_before_writes"]
    assert db.orders["#W5056519"].status == "cancelled"
    assert db.orders["#W5995614"].status == (
        "cancelled" if second_reply == "yes" else "pending"
    )
    if second_reply == "yes":
        assert report["validated_success"]
        assert report["official_result"]["db_check"]["db_match"]
    else:
        assert not report["validated_success"]
    approved = [i for i, m in enumerate(simulation.messages)
                if m.role == "user" and m.content == "yes"]
    writes = [i for i, m in enumerate(simulation.messages)
              if m.role == "assistant" and m.tool_calls
              and m.tool_calls[0].name == "cancel_pending_order"]
    assert len(writes) == len(approved) == expected
    assert all(consent < write for consent, write in zip(approved, writes))
    assert len(agents.contexts) == 7
    assert not agents.steps


def test_fixed_cohorts_partition_train_without_consuming_test(dataset):
    manifest = json.loads((ROOT / "data/manifests/retail_experiment_manifest.json").read_text())
    dev, baseline = manifest["development_task_ids"], manifest["baseline_task_ids"]
    assert len(dev) == len(set(dev)) == 17
    assert len(baseline) == len(set(baseline)) == 57
    assert not set(dev) & set(baseline)
    assert set(dev + baseline) == set(dataset.splits["train"])
    assert not set(dev + baseline) & set(dataset.splits["test"])
    assert manifest["benchmark_commit"] == dataset.manifest["commit"]
    assert all(dataset.load_train_task(task)["initial_state"] is None for task in baseline)


def test_cohort_dry_run_needs_no_api_key_or_model_calls():
    env = {**os.environ, "OPENAI_BASE_URL": "http://127.0.0.1:1"}
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-m", "retail_benchmark", "--cohort", "baseline", "--dry-run"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    listed = json.loads(result.stdout)
    assert len(listed["task_ids"]) == listed["conversations"] == 57
    assert listed["trials"] == 1


def test_changed_request_discards_the_unapproved_queue(adapter, dataset):
    steps = [
        *queued_cancellations(),
        plan(),
        ("order", {"kind": "complete"}),
        policy_review(),
        ("decision", proposal(
            "cancel_pending_order", order_id="#W5995614", reason="ordered by mistake"
        )),
    ]
    graph, agents, config, result = session(
        adapter, dataset, steps,
        "Yara Muller ZIP 85041, no email. Cancel both pending orders: ordered by mistake.",
    )
    assert len(result["queued_actions"]) == 1
    assert adapter.write_count == 0
    result = resume(
        graph, config, "Actually cancel only #W5995614. Keep #W5056519."
    )
    assert adapter.write_count == 0
    assert result["pending_action"].call.arguments["order_id"] == "#W5995614"
    assert result["queued_actions"] == []
    resume(graph, config, "yes")
    db = adapter.snapshot()
    assert adapter.write_count == 1
    assert db.orders["#W5056519"].status == "pending"
    assert db.orders["#W5995614"].status == "cancelled"
    assert not agents.steps
