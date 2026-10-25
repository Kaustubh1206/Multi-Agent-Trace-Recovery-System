"""Serial conversation bridge: official simulator → seven-agent MAS → native graders."""

from datetime import datetime, timezone
from time import monotonic

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator_nl_assertions import (
    DEFAULT_LLM_NL_ASSERTIONS,
    DEFAULT_LLM_NL_ASSERTIONS_ARGS,
)

from retail_mas.graph import build_graph, run_config
from retail_mas.models import AgentModels, describe_model_error
from retail_mas.tools import database_hash

from .evaluation import evaluate, supplementary_checks
from .recording import RecordedAdapter
from .simulator import create_simulator


def now():
    return datetime.now(timezone.utc).isoformat()


def run_task(
    dataset,
    source_task_id,
    *,
    trial=1,
    seed=300,
    max_turns=30,
    user_model=None,
    agents=None,
    on_message=None,
):
    """Run a fresh train trial. Injected agents are only for offline integration tests."""
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    task = Task.model_validate(dataset.load_train_task(source_task_id))
    initial_db = dataset.fresh_db(task.initial_state)
    adapter = RecordedAdapter(initial_db)
    config = run_config(source_task_id)
    config["tags"] = ["retail-mas", "simulated-user"]
    config["metadata"].update(execution_mode="simulated_user", trial=trial, seed=seed)
    started, start_time = monotonic(), now()
    reason, failure, turns = TerminationReason.MAX_STEPS, None, 0
    verifications, paused, stage = [], False, "user"
    scripted = agents is not None
    simulator, models = None, None
    try:
        simulator = create_simulator(task, dataset, user_model, seed)
        user_state = simulator.get_init_state()
        stage = "agent"
        models = agents if scripted else AgentModels()
        graph = build_graph(adapter, models, dataset.policy)
        # Same neutral opening as the upstream half-duplex orchestrator.
        assistant = AssistantMessage(role="assistant", content="Hi! How can I help you today?")
        adapter.trajectory.append(assistant)
        for turns in range(1, max_turns + 1):
            stage = "user"
            reply, user_state = simulator.generate_next_message(assistant, user_state)
            if (
                reply.is_tool_call()
                or not isinstance(reply.content, str)
                or not reply.content.strip()
            ):
                raise ValueError("Retail text simulator must return a nonempty customer message")
            adapter.trajectory.append(reply)
            if on_message:
                on_message("Customer", reply.content)
            if simulator.is_stop(reply):
                reason = TerminationReason.USER_STOP
                break
            stage = "agent"
            update = (
                Command(resume=reply.content)
                if paused
                else {
                    "messages": [HumanMessage(content=reply.content)],
                    "customer_query": reply.content,
                    "new_request": True,
                }
            )
            result = graph.invoke(update, config)
            paused = bool(result.get("__interrupt__"))
            text = result["__interrupt__"][0].value["message"] if paused else result.get("response")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("MAS produced no customer response")
            if not paused and result.get("execution_result") is not None:
                verifications.append(
                    bool((result.get("verification_result") or {}).get("verified"))
                )
            assistant = AssistantMessage(role="assistant", content=text)
            adapter.trajectory.append(assistant)
            if on_message:
                on_message("MAS", text)
    except Exception as error:
        reason = TerminationReason.USER_ERROR if stage == "user" else TerminationReason.AGENT_ERROR
        failure = describe_model_error(error)
    finally:
        if simulator is not None:
            simulator.stop()

    simulation = SimulationRun(
        id=config["configurable"]["thread_id"],
        task_id=source_task_id,
        start_time=start_time,
        end_time=now(),
        duration=monotonic() - started,
        termination_reason=reason,
        messages=adapter.trajectory,
        trial=trial,
        seed=seed,
    )
    grading = evaluate(dataset, task, simulation)
    try:
        checks, changed = supplementary_checks(
            dataset, task, adapter, initial_db, verifications, simulation
        )
    except Exception as error:
        checks, changed = {"supplementary_evaluation_completed": False}, None
        grading.update(status="error", error=describe_model_error(error))
    native = grading["all"]
    if grading["status"] == "complete" and not checks.get("nl_judge_coverage", True):
        grading.update(
            status="invalid_judge_result", error="Native NL judge did not cover every assertion"
        )
    report = {
        "run_id": simulation.id,
        "source_task_id": source_task_id,
        "scenario_id": f"retail-{source_task_id}-trial-{trial}",
        "split": "train",
        "trial": trial,
        "seed": seed,
        "seed_scope": "official_user_simulator",
        "max_customer_turns": max_turns,
        "customer_turns": turns,
        "termination_reason": reason.value,
        "run_error": failure,
        "execution_mode": "offline_scripted_agents" if scripted else "api_models",
        "benchmark_commit": dataset.manifest["commit"],
        "asset_hashes": {key: value["sha256"] for key, value in dataset.manifest["assets"].items()},
        "source_hashes": {key: value["sha256"] for key, value in dataset.manifest["code"].items()},
        "initial_database_hash": database_hash(initial_db),
        "final_database_hash": database_hash(adapter.snapshot()),
        "user_model": simulator.llm if simulator is not None else user_model,
        "user_model_args": simulator.llm_args if simulator is not None else {},
        "mas_model": "scripted" if scripted else getattr(
            getattr(models, "model", None), "model_name", None
        ),
        "judge_model": DEFAULT_LLM_NL_ASSERTIONS,
        "judge_model_args": DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        "evaluation_type": "all",
        "evaluation_status": grading["status"],
        "evaluation_error": grading["error"],
        "official_env_result": grading["env"].model_dump(mode="json") if grading["env"] else None,
        "official_result": native.model_dump(mode="json") if native else None,
        "native_reward": native.reward if native is not None else None,
        "supplementary_checks": checks,
        "changed_records": changed,
        "committed_writes": adapter.write_count,
        "human_transfers": adapter.transfers,
        "guard_rejections": adapter.rejections,
        "validated_success": (
            grading["status"] == "complete"
            and native is not None
            and native.reward == 1.0
            and all(checks.values())
        ),
    }
    return simulation, report, adapter.snapshot()
