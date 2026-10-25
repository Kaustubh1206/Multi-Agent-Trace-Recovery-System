"""Native grading and supplementary checks. Nothing here is available to MAS agents."""

from collections import Counter

from tau2.data_model.simulation import TerminationReason
from tau2.domains.retail.tools import RetailTools
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry

from retail_mas.models import describe_model_error
from retail_mas.nodes import TRANSFER_MESSAGE
from retail_mas.tools import database_hash

DOMAIN = "matrace_pinned_retail"


def pinned_environment(*, dataset, solo_mode=False):
    if solo_mode:
        raise ValueError("Retail does not support solo mode")
    return Environment("retail", dataset.policy, tools=RetailTools(dataset.fresh_db()))


registry.register_domain(pinned_environment, DOMAIN)


def evaluate(dataset, task, simulation):
    """Keep the raw native ENV result even if a later LLM judge request fails."""
    kwargs = dict(
        simulation=simulation,
        task=task,
        solo_mode=False,
        domain=DOMAIN,
        env_kwargs={"dataset": dataset},
        strict_replay=True,
    )
    results = {"env": None, "all": None, "error": None, "status": "complete"}
    for name, kind in (("env", EvaluationType.ENV), ("all", EvaluationType.ALL)):
        try:
            results[name] = evaluate_simulation(evaluation_type=kind, **kwargs)
        except Exception as error:
            results.update(status="error", error=describe_model_error(error))
            break
    if results["all"] is not None:
        simulation.reward_info = results["all"]
    if simulation.termination_reason not in {
        TerminationReason.USER_STOP,
        TerminationReason.AGENT_STOP,
    }:
        results["status"] = "incomplete_run"
    return results


def supplementary_checks(dataset, task, adapter, initial_db, verifications, simulation):
    replay = pinned_environment(dataset=dataset)
    replay.set_state(None, None, adapter.trajectory, strict=True)
    final = adapter.snapshot()
    allowed_orders = {event["order_id"] for event in adapter.commits}
    allowed_users = {event["user_id"] for event in adapter.commits}
    changed = {}
    for group in ("orders", "users", "products"):
        before, after = getattr(initial_db, group), getattr(final, group)
        changed[group] = sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
    requires_transfer = any(
        action.name == "transfer_to_human_agents"
        for action in task.evaluation_criteria.actions or []
    )
    transferred = any(
        message.role == "assistant" and message.content == TRANSFER_MESSAGE
        for message in adapter.trajectory
    )
    checks = {
        "authenticated_before_protected_reads": all(adapter.protected_reads),
        "authenticated_owned_writes": all(c["authenticated_owned"] for c in adapter.commits),
        "exact_confirmation_before_writes": all(c["exact_confirmation"] for c in adapter.commits),
        "no_duplicate_write": len({c["fingerprint"] for c in adapter.commits})
        == adapter.write_count,
        "no_unrelated_record_changes": (
            set(changed["orders"]) <= allowed_orders
            and set(changed["users"]) <= allowed_users
            and not changed["products"]
        ),
        "actual_db_matches_recorded_tools": database_hash(final) == database_hash(replay.tools.db),
        "writes_verified": len(verifications) == adapter.write_count and all(verifications),
        "required_human_handoff": not requires_transfer or (adapter.transfers > 0 and transferred),
    }
    if simulation.reward_info is not None:
        expected = task.evaluation_criteria.nl_assertions or []
        observed = simulation.reward_info.nl_assertions or []
        checks["nl_judge_coverage"] = Counter(expected) == Counter(c.nl_assertion for c in observed)
    return checks, changed
