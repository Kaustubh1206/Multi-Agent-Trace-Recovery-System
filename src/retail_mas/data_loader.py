"""Pinned assets and fresh fixtures. Task records stay with the runner, never agents."""

import hashlib
import json
from pathlib import Path

COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/retail_benchmark_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RetailDataset:
    def __init__(self, root: Path | None = None):
        self.root = root or ROOT / ".cache/tau2-retail"
        self.manifest = json.loads(MANIFEST.read_text())
        for name, expected in self.manifest["assets"].items():
            path = self.root / name
            if not path.exists():
                raise FileNotFoundError("Run: uv run python scripts/fetch_retail_data.py")
            if sha256(path) != expected["sha256"]:
                raise ValueError(f"Pinned asset hash mismatch: {name}")
        self.splits = json.loads((self.root / "split_tasks.json").read_text())
        assert {k: len(v) for k, v in self.splits.items()} == {"train": 74, "test": 40, "base": 114}
        assert set(self.splits["train"]).isdisjoint(self.splits["test"])
        assert set(self.splits["base"]) == set(self.splits["train"] + self.splits["test"])

    @property
    def policy(self) -> str:
        return (self.root / "policy.md").read_text()

    def load_train_task(self, source_task_id: str) -> dict:
        """Runner-only record, including private simulator/evaluator fields."""
        if source_task_id not in self.splits["train"]:
            raise ValueError("Development runner accepts train task IDs only")
        tasks = json.loads((self.root / "tasks.json").read_text())
        return next(task for task in tasks if task["id"] == source_task_id)

    def fresh_db(self, initial_state: dict | None = None):
        from tau2.domains.retail.data_model import RetailDB

        # Fail closed instead of ignoring fixture overrides or dialogue history.
        if initial_state is not None:
            raise NotImplementedError("Non-null task initial_state is not supported in this runner")
        return RetailDB.load(self.root / "db.json")
