"""Run selected training tasks automatically; all trials are independent and serial."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def inspect_reports(directory):
    """Summarize existing artifacts without importing models or rerunning a task."""
    paths = sorted(directory.glob("*.report.json"))
    summary = directory / "summary.json"
    selected = None
    if summary.exists():
        selected = {
            trial["run_id"] for trial in json.loads(summary.read_text()).get("trials", [])
        }
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        if selected is not None and report.get("run_id") not in selected:
            continue
        native = report.get("official_result") or {}
        env = report.get("official_env_result") or {}
        checks = report.get("supplementary_checks") or {}
        row = {
            key: report.get(key) for key in (
                "source_task_id", "trial", "run_id", "termination_reason", "run_error",
                "evaluation_status", "evaluation_error", "run_error_location", "native_reward", "validated_success",
                "committed_writes", "human_transfers", "mas_usage", "mas_call_budget", "architecture", "cohort",
            )
        }
        row.update(
            failed_checks=[name for name, passed in checks.items() if not passed],
            db_check=native.get("db_check") or env.get("db_check"),
            reward_breakdown=native.get("reward_breakdown"),
            failed_nl_assertions=[
                check for check in native.get("nl_assertions") or [] if not check.get("met")
            ],
            failed_communication=[
                check for check in native.get("communicate_checks") or [] if not check.get("met")
            ],
        )
        trajectory = path.with_name(path.name.replace(".report.json", ".simulation.json"))
        if trajectory.exists():
            messages = json.loads(trajectory.read_text()).get("messages") or []
            dialogue = [
                {"role": message["role"], "content": message["content"][:1500]}
                for message in messages
                if message.get("role") in {"user", "assistant"}
                and isinstance(message.get("content"), str)
                and message["content"].strip()
                and not message.get("tool_calls")
            ]
            row["last_dialogue"] = dialogue[-6:]
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No matching trial reports found in this directory")
    return {"trials": rows}


def main():
    parser = argparse.ArgumentParser(description="Official τ² Retail user-simulator pilot")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tasks", nargs="+", help="Train task IDs only")
    mode.add_argument("--inspect", type=Path, help="Inspect saved reports; no model calls")
    mode.add_argument("--cohort", choices=["development", "baseline"], help="Fixed train experiment cohort")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list tasks without API calls")
    parser.add_argument("--trials", type=positive, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-turns", type=positive, default=30)
    parser.add_argument("--user-model", help="OpenAI simulator model; defaults to pinned τ² model")
    parser.add_argument("--output", type=Path, default=Path("work/benchmark"))
    parser.add_argument("--verbose", action="store_true", help="Print actual customer/MAS dialogue")
    args = parser.parse_args()
    if args.inspect is not None:
        print(json.dumps(inspect_reports(args.inspect), indent=2))
        return
    load_dotenv()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from loguru import logger

    logger.disable("tau2")
    from langchain_core.tracers.langchain import wait_for_all_tracers

    from retail_mas.data_loader import RetailDataset

    from .runner import run_task

    dataset = RetailDataset()
    if args.cohort:
        from retail_mas.data_loader import ROOT

        manifest = json.loads((ROOT / "data/manifests/retail_experiment_manifest.json").read_text())
        if manifest["benchmark_commit"] != dataset.manifest["commit"]:
            raise ValueError("Experiment cohort benchmark version mismatch")
        args.tasks = manifest[f"{args.cohort}_task_ids"]
    # Validate every task before making the first paid request; test remains held out.
    for task_id in args.tasks:
        task = dataset.load_train_task(task_id)
        if task["initial_state"] is not None:
            raise NotImplementedError("Non-null task initial_state is not supported")
    if args.dry_run:
        print(json.dumps({"task_ids": args.tasks, "trials": args.trials,
                          "conversations": len(args.tasks) * args.trials}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        for task_id in args.tasks:
            for trial in range(1, args.trials + 1):
                print(f"Task {task_id}, trial {trial}: starting official simulator")
                simulation, report, final_db = run_task(
                    dataset,
                    task_id,
                    trial=trial,
                    seed=args.seed + trial - 1,
                    max_turns=args.max_turns,
                    user_model=args.user_model,
                    on_message=(lambda role, text: print(f"{role}: {text}"))
                    if args.verbose
                    else None,
                )
                report["cohort"] = args.cohort
                stem = f"task-{task_id}-trial-{trial}-{simulation.id}"
                for suffix, content in (
                    ("simulation", simulation.model_dump(mode="json")),
                    ("report", report),
                    ("final_db", final_db.model_dump(mode="json")),
                ):
                    (args.output / f"{stem}.{suffix}.json").write_text(
                        json.dumps(content, indent=2) + "\n"
                    )
                reports.append(
                    {
                        key: report[key]
                        for key in (
                            "run_id",
                            "source_task_id",
                            "trial",
                            "termination_reason",
                            "evaluation_status",
                            "native_reward",
                            "validated_success",
                            "mas_usage",
                        )
                    }
                )
                print(
                    f"  native reward={report['native_reward']}; validated_success={report['validated_success']}; "
                    f"evaluation={report['evaluation_status']}; report={args.output / (stem + '.report.json')}"
                )
                usage = report.get("mas_usage")
                if usage:
                    print(
                        f"  MAS calls={usage['calls']}; input_tokens={usage['input_tokens']}; "
                        f"output_tokens={usage['output_tokens']}; usage_reported={usage['responses_with_usage']}"
                    )
                failed = [name for name, passed in report["supplementary_checks"].items() if not passed]
                print(f"  termination={report['termination_reason']}; failed_checks={failed}")
                if report["run_error"] or report["evaluation_error"]:
                    print(f"  run_error={report['run_error']}; evaluation_error={report['evaluation_error']}")
    finally:
        wait_for_all_tracers()
    (args.output / "summary.json").write_text(json.dumps({"trials": reports}, indent=2) + "\n")
    if any(r["evaluation_status"] != "complete" for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
