import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
# Offline tests must not upload experiment traces even when a local .env enables them.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pytest
from loguru import logger

logger.disable("tau2")

from retail_mas.data_loader import RetailDataset  # noqa: E402

from .oracle import RecordingAdapter, evaluate_environment  # noqa: E402

PILOT_RESULTS = []


def pytest_addoption(parser):
    parser.addoption(
        "--pilot-report", default=None, help="Write actual offline pilot ENV results as JSON"
    )


def pytest_sessionfinish(session, exitstatus):
    import json
    from pathlib import Path

    destination = session.config.getoption("--pilot-report")
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"pytest_exit_status": exitstatus, "pilots": PILOT_RESULTS}, indent=2) + "\n"
        )


@pytest.fixture(scope="session")
def dataset():
    return RetailDataset()


@pytest.fixture
def adapter(dataset):
    return RecordingAdapter(dataset.fresh_db())


@pytest.fixture
def record_pilot(dataset):
    def record(source_task_id, adapter):
        result = evaluate_environment(dataset, source_task_id, adapter)
        assert result["official_raw_result"]["db_check"]["db_match"]
        PILOT_RESULTS.append(result)
        return result

    return record
