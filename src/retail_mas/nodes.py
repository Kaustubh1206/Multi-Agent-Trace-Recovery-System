"""Five specialist LLM roles with deterministic routing and guarded execution."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .models import ModelOutputError
from .objects import AgentStep, PolicyDecision, SupervisorPlan
from .policy import (
    IDENTITY_TOOLS,
    ORDER_TOOLS,
    WRITE_TOOLS,
    asks_for_action_approval,
    confirmation_summary,
    explicit_answer,
)
from .prompts import system_prompt

TRANSFER_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
MAX_STEPS = 32
MAX_TURNS = 40


def context_for(state, adapter, role=None):
    """Only customer dialogue and observed facts; no benchmark task/oracle."""
    return {
        "conversation": [{"role": m.type, "text": m.content} for m in state.get("messages", [])[-8:]],
        "customer_query": state.get("customer_query", ""),
        "authenticated_user_id": adapter.user_id,
        "purpose": state["plan"].purpose if state.get("plan") else "",
        "observations": [] if role == "supervisor" else observed_facts(state),
        "policy_result": state["policy_result"].model_dump()
        if state.get("policy_result")
        else None,
        "validation_error": state.get("error"),
        "handoff_request": state.get("handoff_request", ""),
        "team_write_tools": adapter.supported_write_tools,
        "continuing_request": state.get("continuing_request", False),
        "order_finding": state.get("order_finding", ""),
    }


def observed_facts(state):
    """Keep the latest result for each actual read; never shorten tool result values."""
    import json

    latest = {}
    for observation in state.get("observations", []):
        key = json.dumps(observation["tool"], sort_keys=True)
        latest[key] = observation
    return list(latest.values())


def finish(response, **updates):
    return {
        **updates,
        "response": response,
        "messages": [AIMessage(content=response)],
        "next_agent": "end",
    }


def ask_customer(question, kind="customer"):
    # Checkpoint this question before entering interrupt() on the next Decision visit.
    return {"question": question, "awaiting_customer": kind, "next_agent": "decision"}


def supervisor(state, *, adapter, agents, policy):
    """Plan each customer turn once, then route specialists using their shared results."""
    update = {}
    if state.get("new_request", True):
        adapter.invalidate_consent()
        if state.get("turns", 0) >= MAX_TURNS:
            return finish("Conversation limit reached. Please start a new session.")
        continuing = state.get("continuing_request", False)
        observations = state.get("observations", []) if continuing else []
        handoff_request = state.get("handoff_request", "") if continuing else ""
        clean = {
            **state,
            "observations": observations,
            "plan": state.get("plan") if continuing else None,
            "order_finding": state.get("order_finding", "") if continuing else "",
            "policy_result": None,
            "error": None,
            "handoff_request": handoff_request,
        }
        plan = agents.invoke(
            "supervisor",
            SupervisorPlan,
            system_prompt("supervisor", policy),
            context_for(clean, adapter, "supervisor"),
        )
        if plan.question and asks_for_action_approval(plan.question):
            plan = plan.model_copy(update={"question": ""})
        update = {
            "plan": plan,
            "new_request": False,
            "continuing_request": False,
            "order_finding": state.get("order_finding", "") if continuing else "",
            "observations": observations,
            "policy_result": None,
            "pending_action": None,
            "queued_actions": [],
            "verification_result": None,
            "execution_result": None,
            "confirmation": None,
            "awaiting_customer": None,
            "error": None,
            "handoff_request": handoff_request,
            "steps": 1,
            "turns": state.get("turns", 0) + 1,
        }
        if plan.question:
            return {**update, **ask_customer(plan.question)}
    current = {**state, **update}
    if not adapter.user_id:
        target = "identity"
    else:
        target = "policy" if current["plan"].intent == "unsupported" else "order"
    return {**update, "authenticated_user_id": adapter.user_id, "next_agent": target}


def agent_step(role, state, adapter, agents, policy, tools, allowed):
    context = context_for(state, adapter, role)
    context["available_tools"] = adapter.schemas(tools)
    for _ in range(2):
        step = agents.invoke(role, AgentStep, system_prompt(role, policy), context)
        if step.kind not in allowed:
            raise ValueError(f"Invalid {role} output kind: {step.kind}")
        if step.kind in {"tool", "propose"} and step.tool is None:
            raise ValueError("A tool/proposal step requires a tool call")
        if step.kind not in {"ask", "respond"} or not asks_for_action_approval(step.text):
            return step
        correction = (
            "Do not ask for action approval in prose. Only the application's prepared "
            "action summary can ask approval. "
        )
        correction += {
            "decision": "Return kind=propose with the full official tool call, or "
            "kind=complete to obtain missing observed facts.",
            "order": "Use scoped reads for missing facts; return kind=complete when "
            "order facts are ready. Ask only for genuinely missing customer facts.",
            "identity": "Return a permitted lookup or ask for missing identity facts.",
        }[role]
        context = {**context, "validation_error": correction}
    raise ModelOutputError(f"{role} repeated an unprepared approval request; no action taken")


def read_facts(role, state, adapter, agents, policy, tools):
    """One read per invocation; the agent's self-edge allows further scoped reads."""
    if state.get("steps", 0) >= MAX_STEPS:
        return finish("Unable to complete within the step budget. Please clarify your request.")
    allowed = {"tool", "ask"} if role == "identity" else {"tool", "ask", "complete"}
    step = agent_step(role, state, adapter, agents, policy, tools, allowed)
    update = {"steps": state.get("steps", 0) + 1}
    if step.kind == "ask":
        return {**update, **ask_customer(step.text)}
    if step.kind == "complete":
        return {**update, "order_finding": step.text, "next_agent": "policy"}
    try:
        result = adapter.read(role, step.tool)
        observation = {"tool": step.tool.model_dump(), "result": result}
        target = (
            "policy" if state["plan"].intent == "unsupported" else "order"
        ) if role == "identity" else role
    except (ValueError, TypeError) as error:
        observation = {"tool": step.tool.model_dump(), "error": str(error)}
        target = role
    return {
        **update,
        "observations": state.get("observations", []) + [observation],
        "authenticated_user_id": adapter.user_id,
        "next_agent": target,
    }


def identity_agent(state, *, adapter, agents, policy):
    return read_facts("identity", state, adapter, agents, policy, IDENTITY_TOOLS)


def order_agent(state, *, adapter, agents, policy):
    return read_facts("order", state, adapter, agents, policy, ORDER_TOOLS)


def policy_agent(state, *, adapter, agents, policy):
    """Assess eligibility from the request and observed order facts, without writes."""
    finding = agents.invoke(
        "policy", PolicyDecision, system_prompt("policy", policy), context_for(state, adapter)
    )
    return {"policy_result": finding, "next_agent": "decision"}


def customer_reply(state, adapter):
    """Only this branch pauses; no LLM or retail tool runs again when resumed."""
    question = state["question"]
    kind = state["awaiting_customer"]
    reply = interrupt({"kind": kind, "message": question})
    if not isinstance(reply, str) or not reply.strip():
        adapter.invalidate_consent()
        return finish(
            "No customer reply received; no action taken.",
            pending_action=None,
            queued_actions=[],
            awaiting_customer=None,
        )
    messages = [AIMessage(content=question), HumanMessage(content=reply)]
    update = {"messages": messages, "awaiting_customer": None}
    if kind == "confirmation":
        action = state["pending_action"]
        answer = explicit_answer(reply, action.call.name)
        if answer == "yes" and adapter.confirm(action.fingerprint, reply):
            executed = execute_confirmed(state, adapter)
            return {**update, **executed, "messages": messages + executed.get("messages", [])}
        adapter.invalidate_consent()
        if answer == "no":
            result = finish(
                "No further change made." if state.get("execution_result") else "No change made.",
                pending_action=None, queued_actions=[], confirmation="no", awaiting_customer=None
            )
            return {**result, "messages": messages + result["messages"]}
    return {
        **update,
        "pending_action": None,
        "queued_actions": [],
        "confirmation": None,
        "customer_query": reply,
        "new_request": True,
        "continuing_request": True,
        "handoff_request": (
            "The previous reply did not give unqualified approval. Resolve the customer's "
            "questions, changed details or conditions before asking again. Explain unsupported "
            "terms rather than silently repeating the proposal; never promise an unsupported deadline."
            if kind == "confirmation" else ""
        ),
        "next_agent": "supervisor",
    }


def decision_agent(state, *, adapter, agents, policy):
    """Propose the allowed action and obtain exact customer consent before execution."""
    if state.get("awaiting_customer"):
        return customer_reply(state, adapter)
    if state.get("steps", 0) >= MAX_STEPS:
        return finish("Unable to complete within the step budget. Please clarify your request.")
    step = agent_step(
        "decision",
        state,
        adapter,
        agents,
        policy,
        WRITE_TOOLS,
        {"ask", "respond", "complete", "propose", "transfer"},
    )
    update = {"steps": state.get("steps", 0) + 1}
    if step.kind == "ask":
        return {**update, **ask_customer(step.text)}
    if step.kind == "respond":
        return finish(step.text, **update)
    if step.kind == "transfer":
        adapter.invalidate_consent()
        adapter.transfer(step.text)
        return finish(TRANSFER_MESSAGE, **update)
    if step.kind == "complete":
        return {
            **update,
            "handoff_request": step.text,
            "policy_result": None,
            "next_agent": "order",
        }
    try:
        if not state["policy_result"].allowed:
            raise ValueError(state["policy_result"].reason)
        action = adapter.prepare(step.tool)
        return {
            **update,
            "pending_action": action,
            "queued_actions": step.following_actions,
            "handoff_request": "",
            "error": None,
            **ask_customer(confirmation_summary(action), "confirmation"),
        }
    except (ValueError, TypeError) as error:
        return {**update, "error": str(error), "pending_action": None, "next_agent": "decision"}



def action_response(action, checked):
    """Describe the observed simulated change without another model call."""
    target = action.call.arguments.get("order_id", action.call.arguments.get("user_id"))
    status = checked["record"].get("status", "updated")
    response = f"{action.call.name} completed for {target}. Status: {status}."
    if action.call.name == "return_delivered_order_items":
        effect = action.financial_effect
        response += (
            f" Requested refund: ${effect['refund_after_return']:,.2f}."
            f" Amount paid for remaining items: ${effect['remaining_items_paid']:,.2f}."
            f" {effect['refund_timing']}"
        )
    if status in {"return requested", "exchange requested"}:
        response += " You will receive email instructions for returning the items."
    return response


def execute_confirmed(state, adapter):
    """Execute and verify one exact approved call; later queued calls need new consent."""
    action = state["pending_action"]
    try:
        result = adapter.execute(action)
    except (ValueError, TypeError) as error:
        return finish(f"Action could not be completed: {error}", error=str(error),
                      pending_action=None, queued_actions=[], confirmation=None)
    try:
        checked = adapter.verify(action, result)
    except Exception:
        checked = {"matches_confirmed_action": False}
    verified = checked["matches_confirmed_action"]
    update = {
        "execution_result": result,
        "observations": [],
        "order_finding": "",
        "verification_result": {**checked, "verified": verified},
        "pending_action": None,
        "confirmation": None,
        "queued_actions": [],
    }
    if not verified:
        return finish(
            "The change was committed, but verification did not pass. Human review is needed; "
            "do not repeat the action.", **update
        )
    response = action_response(action, checked)
    remaining = state.get("queued_actions", [])
    if remaining:
        try:
            next_action = adapter.prepare(remaining[0])
        except (ValueError, TypeError) as error:
            return finish(response + f" The next request needs clarification: {error}", **update)
        return {
            **update,
            "pending_action": next_action,
            "queued_actions": remaining[1:],
            **ask_customer(response + "\n\n" + confirmation_summary(next_action), "confirmation"),
        }
    return finish(response, **update)
