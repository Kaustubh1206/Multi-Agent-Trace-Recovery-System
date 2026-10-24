"""Generate draft evaluator-only labels from pinned reference operations.

Run from repository root: uv run python -m tests.build_labels
Passing reference tools is NOT human label validation or an MAS benchmark score.
"""

import hashlib
import json
import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def main():
    from loguru import logger

    logger.disable("tau2")
    from tau2.domains.retail.tools import RetailTools

    from retail_mas.data_loader import ROOT, RetailDataset

    dataset = RetailDataset()
    records = []
    for task_id in dataset.manifest["pilot_train_ids"]:
        task = dataset.load_train_task(task_id)
        db = dataset.fresh_db(task["initial_state"])
        tools = RetailTools(db)
        errors = []
        for action in task["evaluation_criteria"]["actions"]:
            try:
                tools.use_tool(action["name"], **action["arguments"])
            except ValueError as error:
                errors.append({"action_id": action["action_id"], "error": str(error)})
                if task_id != "47" or action["name"] != "get_order_details":
                    raise
        canonical = json.dumps(db.model_dump(mode="json"), sort_keys=True)
        records.append(
            {
                "source_task_id": task_id,
                "scenario_id": "pilot-" + task_id,
                "split": "train",
                "benchmark_ref": "v1.0.1",
                "benchmark_commit": dataset.manifest["commit"],
                "initial_state": task["initial_state"],
                "initial_database_hash": dataset.manifest["assets"]["db.json"]["sha256"],
                "policy_hash": dataset.manifest["assets"]["policy.md"]["sha256"],
                "fixture_reference": ".cache/tau2-retail/db.json",
                "official_evaluation_criteria": task["evaluation_criteria"],
                "expected_database_canonical_sha256": hashlib.sha256(
                    canonical.encode()
                ).hexdigest(),
                "reference_errors": errors,
                "expected_effect": "human_handoff_without_db_write"
                if task_id == "50"
                else "reference_db_outcome",
                "required_preconditions": [
                    "authenticate before protected activity",
                    "ownership",
                    "current eligibility",
                    "exact explicit consent before write",
                ],
                "forbidden_effects": [
                    "unrelated record changes",
                    "duplicate write",
                    "unconfirmed write",
                ],
                "communication_expectations": task["evaluation_criteria"].get("nl_assertions")
                or [],
                "label_review_status": "draft",
                "review_note": "Reference outcome computed automatically. Independent human review pending; communication completeness is not validated by this label.",
            }
        )
    destination = ROOT / "data/ground_truth/clean_tasks.jsonl"
    destination.write_text("".join(json.dumps(record) + "\n" for record in records))
    print(f"Wrote {len(records)} draft labels; no completed benchmark trials claimed.")


if __name__ == "__main__":
    main()
