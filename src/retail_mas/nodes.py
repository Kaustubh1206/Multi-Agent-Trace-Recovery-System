"""Small specialist nodes and deterministic tool/consent boundaries."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .objects import (
    AgentStep,
    HandoffResult,
    PaymentFinding,
    PolicyDecision,
    SupervisorPlan,
    ToolCall,
)
from .policy import IDENTITY_TOOLS, ORDER_TOOLS, WRITE_TOOLS, confirmation_summary, explicit_answer
from .prompts import system_prompt

TRANSFER_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
MAX_STEPS = 32
MAX_TURNS = 40


def context_for(state, adapter):
    """Explicit allowlist: no task records, oracle labels or full database."""
    return {
        "conversation": [{"role": m.type, "text": m.content} for m in state.get("messages", [])],
        "customer_query": state.get("customer_query", ""),
        "authenticated_user_id": adapter.user_id,
        "purpose": state["plan"].purpose if state.get("plan") else "",
        "observations": state.get("observations", []),
        "validation_error": state.get("error"),
        "handoff_request": state["step"].text if state.get("step") else "",
    }


def handoff(source, target, purpose):
    return HandoffResult(from_agent=source, to_agent=target, purpose=purpose)


def supervisor(state, *, adapter, agents, policy):
    adapter.invalidate_consent()
    if state.get("turns", 0) >= MAX_TURNS:
        return {
            "next_node": "stop",
            "response": "Conversation limit reached. Please start a new session.",
        }
    # A new customer turn invalidates every subtask proposal, result and observation.
    clean = {**state, "observations": [], "plan": None, "step": None, "error": None}
    plan = agents.invoke(
        "supervisor",
        SupervisorPlan,
        system_prompt("supervisor", policy),
        context_for(clean, adapter),
    )
    target = "ask" if plan.question else "identity" if not adapter.user_id else "order"
    if adapter.user_id and plan.intent == "unsupported" and not plan.question:
        target = "decision"
    return {
        "plan": plan,
        "next_node": target,
        "response": plan.question,
        "steps": 1,
        "turns": state.get("turns", 0) + 1,
        "observations": [],
        "error": None,
        "pending_action": None,
        "policy_result": None,
        "payment_info": None,
        "execution_result": None,
        "confirmation": None,
        "step": None,
        "authenticated_user_id": adapter.user_id,
        "handoff": handoff("supervisor", target, plan.purpose),
    }


def specialist(state, *, role, adapter, agents, policy):
    if state.get("steps", 0) >= MAX_STEPS:
        return {
            "next_node": "stop",
            "response": "Unable to complete within the step budget. Please clarify your request.",
        }
    context = context_for(state, adapter)
    scopes = {"identity": IDENTITY_TOOLS, "order": ORDER_TOOLS, "decision": WRITE_TOOLS}
    context["available_tools"] = adapter.schemas(scopes[role])
    step = agents.invoke(role, AgentStep, system_prompt(role, policy), context)
    allowed = {
        "identity": {"tool", "ask"},
        "order": {"tool", "ask", "complete"},
        "decision": {"ask", "respond", "complete", "propose", "transfer"},
    }
    if step.kind not in allowed[role]:
        raise ValueError(f"Invalid {role} output kind: {step.kind}")
    target = {
        "tool": "read",
        "ask": "ask",
        "complete": "decision",
        "respond": "stop",
        "propose": "prepare",
        "transfer": "transfer",
    }[step.kind]
    if role == "decision" and step.kind == "complete":
        target = "order"
    if step.kind in {"tool", "propose"} and step.tool is None:
        raise ValueError("A tool/proposal step requires a tool call")
    return {
        "step": step,
        "active_agent": role,
        "next_node": target,
        "response": step.text,
        "steps": state.get("steps", 0) + 1,
        "handoff": handoff(role, target, step.kind),
    }


def read_tool(state, *, adapter):
    role = state["active_agent"]
    call = state["step"].tool
    try:
        result = adapter.read(role, call)
        observation = {"tool": call.model_dump(), "result": result}
        target = "order" if role == "identity" else role
    except (ValueError, TypeError) as error:
        observation = {"tool": call.model_dump(), "error": str(error)}
        target = role
    return {
        "observations": state.get("observations", []) + [observation],
        "next_node": target,
        "authenticated_user_id": adapter.user_id,
    }


def prepare_action(state, *, adapter):
    try:
        action = adapter.prepare(state["step"].tool)
        return {"pending_action": action, "next_node": "policy", "error": None}
    except (ValueError, TypeError) as error:
        return {"pending_action": None, "next_node": "decision", "error": str(error)}


def policy_agent(state, *, adapter, agents, policy):
    context = context_for(state, adapter)
    context.update(proposal=state["pending_action"].model_dump(), deterministic_validation="passed")
    finding = agents.invoke("policy", PolicyDecision, system_prompt("policy", policy), context)
    if not finding.allowed:
        adapter.invalidate_consent()
        return {"policy_result": finding, "response": finding.reason, "next_node": "stop"}
    action = state["pending_action"]
    relevant = (
        "payment_method_id" in action.call.arguments or action.call.name == "cancel_pending_order"
    )
    return {"policy_result": finding, "next_node": "payment" if relevant else "confirm"}


def payment_agent(state, *, adapter, agents, policy):
    context = context_for(state, adapter)
    profile = adapter.read(
        "payment", ToolCall(name="get_user_details", arguments={"user_id": adapter.user_id})
    )
    context["allowed_payment_methods"] = profile["payment_methods"]
    context["validated_financial_effect"] = state["pending_action"].financial_effect
    finding = agents.invoke("payment", PaymentFinding, system_prompt("payment", policy), context)
    return {"payment_info": finding.model_dump(), "next_node": "confirm"}


def ask_customer(state):
    question = state.get("response") or "Please clarify your request."
    reply = interrupt({"kind": "customer", "message": question})
    if not isinstance(reply, str) or not reply.strip():
        return {"next_node": "stop", "response": "No customer reply received; no action taken."}
    return {
        "messages": [AIMessage(content=question), HumanMessage(content=reply)],
        "customer_query": reply,
        "next_node": "supervisor",
    }


def confirm_action(state, *, adapter):
    action = state["pending_action"]
    summary = confirmation_summary(action)
    # interrupt occurs BEFORE consent or writes; replaying this node is safe.
    reply = interrupt(
        {"kind": "confirmation", "message": summary, "fingerprint": action.fingerprint}
    )
    if not isinstance(reply, str):
        reply = ""
    answer = explicit_answer(reply)
    update = {
        "messages": [AIMessage(content=summary), HumanMessage(content=reply)],
        "confirmation": answer,
    }
    if answer == "yes" and adapter.confirm(action.fingerprint, reply):
        return {**update, "next_node": "execute"}
    adapter.invalidate_consent()
    if answer == "no":
        return {
            **update,
            "pending_action": None,
            "next_node": "stop",
            "response": "No change made.",
        }
    # Ambiguity AND changes of mind both return to planning; no stale consent survives.
    return {**update, "pending_action": None, "customer_query": reply, "next_node": "supervisor"}


def execute_action(state, *, adapter):
    action = state["pending_action"]
    try:
        result = adapter.execute(action)
        status = result.get("status", "updated")
        target = action.call.arguments.get("order_id", action.call.arguments.get("user_id"))
        response = f"{action.call.name} completed for {target}. Status: {status}."
        if status in {"return requested", "exchange requested"}:
            response += " You will receive email instructions for returning the items."
        return {
            "execution_result": result,
            "response": response,
            "next_node": "stop",
            "pending_action": None,
            "confirmation": None,
        }
    except (ValueError, TypeError) as error:
        return {
            "error": str(error),
            "response": f"No change made: {error}",
            "next_node": "stop",
            "pending_action": None,
            "confirmation": None,
        }


def transfer_customer(state, *, adapter):
    adapter.invalidate_consent()
    adapter.transfer(state["step"].text)
    return {"response": TRANSFER_MESSAGE, "next_node": "stop"}


def respond(state):
    return {"messages": [AIMessage(content=state["response"])]}
