"""The seven agents in the proposal. Tool checks stay inside the adapter."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .objects import (
    AgentStep,
    PaymentFinding,
    PolicyDecision,
    SupervisorPlan,
    ToolCall,
    VerificationFinding,
)
from .policy import IDENTITY_TOOLS, ORDER_TOOLS, WRITE_TOOLS, confirmation_summary, explicit_answer
from .prompts import system_prompt

TRANSFER_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
MAX_STEPS = 32
MAX_TURNS = 40


def context_for(state, adapter):
    """Only customer dialogue and observed facts; no benchmark task/oracle."""
    return {
        "conversation": [{"role": m.type, "text": m.content} for m in state.get("messages", [])],
        "customer_query": state.get("customer_query", ""),
        "authenticated_user_id": adapter.user_id,
        "purpose": state["plan"].purpose if state.get("plan") else "",
        "observations": state.get("observations", []),
        "policy_result": state["policy_result"].model_dump()
        if state.get("policy_result")
        else None,
        "validation_error": state.get("error"),
        "handoff_request": state.get("handoff_request", ""),
    }


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
        clean = {
            **state,
            "observations": [],
            "plan": None,
            "policy_result": None,
            "error": None,
            "handoff_request": "",
        }
        plan = agents.invoke(
            "supervisor",
            SupervisorPlan,
            system_prompt("supervisor", policy),
            context_for(clean, adapter),
        )
        update = {
            "plan": plan,
            "new_request": False,
            "order_complete": False,
            "observations": [],
            "policy_result": None,
            "pending_action": None,
            "payment_info": None,
            "verification_result": None,
            "execution_result": None,
            "confirmation": None,
            "awaiting_customer": None,
            "error": None,
            "handoff_request": "",
            "steps": 1,
            "turns": state.get("turns", 0) + 1,
        }
        if plan.question:
            return {**update, **ask_customer(plan.question)}
    current = {**state, **update}
    if not adapter.user_id:
        target = "identity"
    elif not current.get("order_complete") and current["plan"].intent != "unsupported":
        target = "order"
    elif current.get("policy_result") is None:
        target = "policy"
    else:
        target = "decision"
    return {**update, "authenticated_user_id": adapter.user_id, "next_agent": target}


def agent_step(role, state, adapter, agents, policy, tools, allowed):
    context = context_for(state, adapter)
    context["available_tools"] = adapter.schemas(tools)
    step = agents.invoke(role, AgentStep, system_prompt(role, policy), context)
    if step.kind not in allowed:
        raise ValueError(f"Invalid {role} output kind: {step.kind}")
    if step.kind in {"tool", "propose"} and step.tool is None:
        raise ValueError("A tool/proposal step requires a tool call")
    return step


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
        return {**update, "order_complete": True, "next_agent": "supervisor"}
    try:
        result = adapter.read(role, step.tool)
        observation = {"tool": step.tool.model_dump(), "result": result}
        target = "supervisor" if role == "identity" else role
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
    return {"policy_result": finding, "next_agent": "supervisor"}


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
            awaiting_customer=None,
        )
    messages = [AIMessage(content=question), HumanMessage(content=reply)]
    update = {"messages": messages, "awaiting_customer": None}
    if kind == "confirmation":
        action = state["pending_action"]
        answer = explicit_answer(reply, action.call.name)
        if answer == "yes" and adapter.confirm(action.fingerprint, reply):
            return {**update, "confirmation": answer, "next_agent": "action_payment"}
        adapter.invalidate_consent()
        if answer == "no":
            result = finish(
                "No change made.", pending_action=None, confirmation="no", awaiting_customer=None
            )
            return {**result, "messages": messages + result["messages"]}
    return {
        **update,
        "pending_action": None,
        "confirmation": None,
        "customer_query": reply,
        "new_request": True,
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
            "order_complete": False,
            "policy_result": None,
            "next_agent": "supervisor",
        }
    try:
        if not state["policy_result"].allowed:
            raise ValueError(state["policy_result"].reason)
        action = adapter.prepare(step.tool)
        return {
            **update,
            "pending_action": action,
            "error": None,
            **ask_customer(confirmation_summary(action), "confirmation"),
        }
    except (ValueError, TypeError) as error:
        return {**update, "error": str(error), "pending_action": None, "next_agent": "decision"}


def action_payment_agent(state, *, adapter, agents, policy):
    """Review payment facts, then use the deterministic executor for the approved write."""
    action = state["pending_action"]
    context = context_for(state, adapter)
    profile = adapter.read(
        "action_payment", ToolCall(name="get_user_details", arguments={"user_id": adapter.user_id})
    )
    context.update(
        approved_action=action.call.model_dump(),
        allowed_payment_methods=profile["payment_methods"],
        validated_financial_effect=action.financial_effect,
    )
    finding = agents.invoke(
        "action_payment", PaymentFinding, system_prompt("action_payment", policy), context
    )
    try:
        result = adapter.execute(action)
        return {
            "payment_info": finding.model_dump(),
            "execution_result": result,
            "next_agent": "verification",
        }
    except (ValueError, TypeError) as error:
        return finish(
            f"No change made: {error}", error=str(error), pending_action=None, confirmation=None
        )


def verification_agent(state, *, adapter, agents, policy):
    """Check the committed DB against the confirmed action; never load benchmark answers."""
    action = state["pending_action"]
    checked = adapter.verify(action, state["execution_result"])
    context = context_for(state, adapter)
    context.update(confirmed_action=action.call.model_dump(), observed_outcome=checked)
    finding = agents.invoke(
        "verification", VerificationFinding, system_prompt("verification", policy), context
    )
    verified = checked["matches_confirmed_action"] and finding.verified
    if verified:
        target = action.call.arguments.get("order_id", action.call.arguments.get("user_id"))
        status = checked["record"].get("status", "updated")
        response = f"{action.call.name} completed for {target}. Status: {status}."
        if status in {"return requested", "exchange requested"}:
            response += " You will receive email instructions for returning the items."
    else:
        response = "The change was committed, but verification did not pass. Human review is needed; do not repeat the action."
    return finish(
        response,
        verification_result={**finding.model_dump(), **checked, "verified": verified},
        pending_action=None,
        confirmation=None,
    )
