"""Local Studio entry point. Each thread owns one in-memory simulated retail session."""

import asyncio
from dataclasses import dataclass, field
from functools import partial
from threading import Lock
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph_sdk.runtime import ServerRuntime
from pydantic import BaseModel

from .data_loader import RetailDataset
from .graph import build_graph
from .models import AgentModels
from .objects import PendingAction, PolicyDecision, SupervisorPlan, ToolCall
from .tools import RetailAdapter


@dataclass
class Session:
    adapter: RetailAdapter
    agents: AgentModels
    policy: str
    token: str = field(default_factory=lambda: str(uuid4()))


_sessions: dict[str, Session] = {}
_lock = Lock()


def get_session(thread_id: str) -> Session:
    """Reuse the same DB and consent across turns/resumes, never across threads."""
    with _lock:
        if thread_id not in _sessions:
            dataset = RetailDataset()
            _sessions[thread_id] = Session(
                RetailAdapter(dataset.fresh_db()), AgentModels(), dataset.policy
            )
        return _sessions[thread_id]


def session_node(function, state, *, session):
    if state.get("studio_session_id", session.token) != session.token:
        raise ValueError("Retail session expired after server restart. Create a new Studio thread.")
    # Store plain JSON in server checkpoints; restore validated objects only inside nodes.
    current = dict(state)
    for key, schema in (
        ("plan", SupervisorPlan),
        ("policy_result", PolicyDecision),
        ("pending_action", PendingAction),
    ):
        if current.get(key) is not None:
            current[key] = schema.model_validate(current[key])
    current["queued_actions"] = [
        ToolCall.model_validate(call) for call in current.get("queued_actions", [])
    ]
    update = function(current)
    return {
        **{
            key: value.model_dump() if isinstance(value, BaseModel) else value
            for key, value in update.items()
        },
        "studio_session_id": session.token,
    }


async def make_graph(config: RunnableConfig, runtime: ServerRuntime):
    """Agent Server calls this for runs, graph rendering, schemas and state reads."""
    if runtime.execution_runtime is None:
        # Rendering the graph needs neither credentials nor a retail DB session.
        return build_graph(None, None, "", server=True)
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        raise ValueError("Create a Studio thread before running the Retail MAS.")
    session = await asyncio.to_thread(get_session, str(thread_id))
    return build_graph(
        session.adapter,
        session.agents,
        session.policy,
        server=True,
        wrap_node=lambda function: partial(session_node, function, session=session),
    )
