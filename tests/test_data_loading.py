import inspect
import json
from pathlib import Path

import pytest
import tau2.domains.retail.data_model as retail_model
import tau2.domains.retail.environment as retail_environment
import tau2.domains.retail.tools as retail_tools

from retail_mas.data_loader import COMMIT, sha256


def test_pin_split_and_pilots(dataset):
    assert dataset.manifest["commit"] == COMMIT
    for task_id in ["0", "1", "47", "50", "69", "84"]:
        task = dataset.load_train_task(task_id)
        assert task["id"] == task_id
        assert task["initial_state"] is None
        assert task["user_scenario"]["instructions"]
        assert task["evaluation_criteria"]


def test_installed_official_sources_match_pin(dataset):
    for module in [retail_model, retail_environment, retail_tools]:
        path = Path(inspect.getfile(module))
        reference = "src/tau2/domains/retail/" + path.name
        assert sha256(path) == dataset.manifest["code"][reference]["sha256"]


def test_test_split_is_held_out(dataset):
    with pytest.raises(ValueError, match="train"):
        dataset.load_train_task(dataset.splits["test"][0])


def test_fresh_db_and_unsupported_initial_state(dataset):
    first = dataset.fresh_db()
    first.orders["#W2417020"].status = "cancelled"
    assert dataset.fresh_db().orders["#W2417020"].status == "pending"
    with pytest.raises(NotImplementedError):
        dataset.fresh_db({"initialization_data": {}})


def test_task_69_is_return_then_cancellation(dataset):
    task = dataset.load_train_task("69")
    assert "return" in task["user_scenario"]["instructions"]["reason_for_call"]
    assert task["evaluation_criteria"]["actions"][-1]["name"] == "cancel_pending_order"


def test_assets_detect_tampering(dataset, tmp_path):
    from retail_mas.data_loader import RetailDataset

    for name in dataset.manifest["assets"]:
        (tmp_path / name).write_bytes((dataset.root / name).read_bytes())
    (tmp_path / "db.json").write_text(json.dumps({"users": {}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        RetailDataset(tmp_path)
