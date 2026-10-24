import ast
import json
from pathlib import Path

from .support import cancellation_steps, resume, session


def test_private_task_fields_never_reach_any_agent(adapter, dataset):
    task = dataset.load_train_task("69")
    private_text = task["user_scenario"]["instructions"]["task_instructions"]
    graph, agents, config, _ = session(
        adapter,
        dataset,
        cancellation_steps(),
        "Emma Smith ZIP 10192, no email; cancel #W2417020 because no longer needed.",
        "69",
    )
    result = resume(graph, config, "yes")
    # All seven roles were actually invoked and checked, including the policy/payment roles.
    assert {role for role, _, _ in agents.contexts} == {
        "supervisor",
        "identity",
        "order",
        "decision",
        "policy",
        "action_payment",
        "verification",
    }
    serialized = json.dumps(agents.contexts)
    for forbidden in [
        "evaluation_criteria",
        "user_scenario",
        "reason_for_call",
        "known_info",
        "unknown_info",
        "task_instructions",
        "label_review_status",
        private_text,
    ]:
        assert forbidden not in serialized
        assert forbidden not in result
    assert "source_task_id" not in serialized


def test_runtime_never_imports_test_oracle_or_reads_labels():
    root = Path(__file__).resolve().parents[1] / "src/retail_mas"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("tests", "tau2.evaluator"))
            if isinstance(node, ast.Import):
                assert all(
                    not name.name.startswith(("tests", "tau2.evaluator")) for name in node.names
                )
        assert "clean_tasks.jsonl" not in path.read_text()
        assert "ground_truth/" not in path.read_text()


def test_labels_preserve_source_and_review_status(dataset):
    path = Path(__file__).resolve().parents[1] / "data/ground_truth/clean_tasks.jsonl"
    labels = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row["source_task_id"] for row in labels} == set(dataset.manifest["pilot_train_ids"])
    for row in labels:
        original = dataset.load_train_task(row["source_task_id"])
        assert row["official_evaluation_criteria"] == original["evaluation_criteria"]
        assert row["initial_state"] == original["initial_state"]
        assert row["label_review_status"] == "draft"
        assert row["benchmark_commit"] == dataset.manifest["commit"]
        assert "root_cause_agent_id" not in row
