"""TEST ONLY: official evaluator and expected-state construction. Never agent input."""

import json

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau2.domains.retail.tools import RetailTools
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

from retail_mas.tools import RetailAdapter


class RecordingAdapter(RetailAdapter):
    """Evaluator-only capture of actual committed tool calls, excluding preflight copies."""

    def __init__(self, db):
        super().__init__(db)
        self.trajectory = []

    def record(self, name, arguments, result):
        tool_call = ToolCall(id=str(len(self.trajectory)), name=name, arguments=arguments)
        self.trajectory.extend(
            [
                AssistantMessage(role="assistant", tool_calls=[tool_call]),
                ToolMessage(role="tool", id=tool_call.id, content=json.dumps(result)),
            ]
        )

    def execute(self, action):
        result = super().execute(action)
        self.record(action.call.name, action.call.arguments, result)
        return result

    def transfer(self, summary):
        result = super().transfer(summary)
        self.record("transfer_to_human_agents", {"summary": summary}, result)
        return result


def evaluate_environment(dataset, source_task_id, adapter):
    task = Task.model_validate(dataset.load_train_task(source_task_id))

    def environment(solo_mode=False):
        assert not solo_mode
        return Environment(
            domain_name="retail",
            policy=dataset.policy,
            tools=RetailTools(dataset.fresh_db(task.initial_state)),
        )

    raw = EnvironmentEvaluator.calculate_reward(environment, task, adapter.trajectory)
    # Independently compare the actual in-memory DB, beyond replay of tool messages.
    expected = environment()
    for action in task.evaluation_criteria.actions:
        try:
            expected.make_tool_call(action.name, action.requestor, **action.arguments)
        except ValueError:
            # Task 47 deliberately includes invalid early order references.
            if source_task_id != "47" or action.name != "get_order_details":
                raise
    assert adapter.snapshot() == expected.tools.db
    return {
        "source_task_id": source_task_id,
        "run_type": "offline_scripted_llm_real_tools",
        "official_evaluation_type": "ENV only",
        "official_raw_result": raw.model_dump(mode="json"),
        "actual_db_matches_reference": True,
        "committed_writes": adapter.write_count,
        "simulated_user_run": False,
        "full_benchmark_score": None,
    }
