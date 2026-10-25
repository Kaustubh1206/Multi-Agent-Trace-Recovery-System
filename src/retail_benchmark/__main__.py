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


def main():
    parser = argparse.ArgumentParser(description="Official τ² Retail user-simulator pilot")
    parser.add_argument("--tasks", nargs="+", required=True, help="Train task IDs only")
    parser.add_argument("--trials", type=positive, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-turns", type=positive, default=30)
    parser.add_argument("--user-model", help="OpenAI simulator model; defaults to pinned τ² model")
    parser.add_argument("--output", type=Path, default=Path("work/benchmark"))
    parser.add_argument("--verbose", action="store_true", help="Print actual customer/MAS dialogue")
    args = parser.parse_args()
    load_dotenv()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from loguru import logger

    logger.disable("tau2")
    from langchain_core.tracers.langchain import wait_for_all_tracers

    from retail_mas.data_loader import RetailDataset

    from .runner import run_task

    dataset = RetailDataset()
    # Validate every task before making the first paid request; test remains held out.
    for task_id in args.tasks:
        task = dataset.load_train_task(task_id)
        if task["initial_state"] is not None:
            raise NotImplementedError("Non-null task initial_state is not supported")
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
                        )
                    }
                )
                print(
                    f"  native reward={report['native_reward']}; checks passed={report['validated_success']}; "
                    f"evaluation={report['evaluation_status']}; report={args.output / (stem + '.report.json')}"
                )
    finally:
        wait_for_all_tracers()
    (args.output / "summary.json").write_text(json.dumps({"trials": reports}, indent=2) + "\n")
    if any(r["evaluation_status"] != "complete" for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
