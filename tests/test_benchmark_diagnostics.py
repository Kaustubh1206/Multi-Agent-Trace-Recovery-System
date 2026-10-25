"""Saved-run diagnosis must not invoke models or mix old batches into current results."""

import json
import os
import subprocess
import sys
from pathlib import Path

from retail_benchmark.__main__ import inspect_reports


def save_report(directory, run_id, **updates):
    report = {
        "run_id": run_id,
        "source_task_id": "69",
        "trial": 1,
        "termination_reason": "agent_error",
        "run_error": "ModelOutputError",
        "evaluation_status": "incomplete_run",
        "evaluation_error": None,
        "native_reward": 0.0,
        "validated_success": False,
        "supplementary_checks": {"no_duplicate_write": True},
        "official_result": None,
        "official_env_result": {"db_check": {"db_match": False, "db_reward": 0.0}},
        **updates,
    }
    path = directory / f"task-{run_id}.report.json"
    path.write_text(json.dumps(report))
    return path


def test_inspector_selects_batch_and_excludes_private_metadata(tmp_path):
    save_report(tmp_path, "old")
    current = save_report(tmp_path, "current")
    (tmp_path / "summary.json").write_text(json.dumps({"trials": [{"run_id": "current"}]}))
    current.with_name(current.name.replace(".report.json", ".simulation.json")).write_text(
        json.dumps({"messages": [
            {"role": "user", "content": "Cancel my pending order.",
             "raw_data": {"private": "provider-metadata-sentinel"}},
            {"role": "assistant", "content": None,
             "tool_calls": [{"name": "get_order_details"}]},
            {"role": "tool", "content": "protected-tool-record-sentinel"},
            {"role": "assistant", "content": "Please confirm."},
        ]})
    )
    result = inspect_reports(tmp_path)
    assert len(result["trials"]) == 1
    row = result["trials"][0]
    assert row["run_id"] == "current"
    assert row["termination_reason"] == "agent_error"
    assert row["db_check"]["db_match"] is False
    # Native failure is not mislabeled as a failed safety check.
    assert row["validated_success"] is False
    assert row["failed_checks"] == []
    assert [m["role"] for m in row["last_dialogue"]] == ["user", "assistant"]
    assert "provider-metadata-sentinel" not in json.dumps(result)
    assert "protected-tool-record-sentinel" not in json.dumps(result)


def test_inspect_cli_works_without_key_and_preserves_reports(tmp_path):
    report_path = save_report(
        tmp_path, "score-zero",
        termination_reason="user_stop", run_error=None, evaluation_status="complete",
        supplementary_checks={"required_human_handoff": False},
        official_result={
            "db_check": {"db_match": False, "db_reward": 0.0},
            "reward_breakdown": {"DB": 0.0, "NL_ASSERTION": 0.0},
            "nl_assertions": [{"nl_assertion": "Refund amount communicated", "met": False}],
            "communicate_checks": [{"info": "1095.55", "met": False}],
        },
    )
    before = report_path.read_bytes()
    completed = subprocess.run(
        [sys.executable, "-m", "retail_benchmark", "--inspect", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OPENAI_API_KEY": "", "LANGSMITH_TRACING": "false",
             "LANGCHAIN_TRACING_V2": "false"},
        capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    row = json.loads(completed.stdout)["trials"][0]
    assert row["failed_checks"] == ["required_human_handoff"]
    assert len(row["failed_nl_assertions"]) == 1
    assert len(row["failed_communication"]) == 1
    assert report_path.read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 1
