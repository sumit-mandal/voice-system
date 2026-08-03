"""Email LangGraph: classify → route → tools → draft reply."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.agent.llm import chat_completion
from app.email_agent.prompts import (
    CLASSIFY_SYSTEM,
    DRAFT_SYSTEM,
    build_classify_prompt,
    build_draft_prompt,
)
from app.email_agent.state import EmailState
from app.email_agent.tools.records import fetch_history_records
from app.logging_setup import get_logger

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_VALID_INTENTS: set[str] = {
    "history_records",
    "appointment",
    "billing",
    "general_question",
    "unclear",
    "escalate",
}


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def classify_intent(state: EmailState) -> EmailState:
    raw = chat_completion(
        [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {
                "role": "user",
                "content": build_classify_prompt(
                    subject=state["subject"],
                    body_text=state["body_text"],
                    from_address=state["from_address"],
                ),
            },
        ]
    )
    data = _parse_json(raw)
    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        intent = "unclear"

    return {
        **state,
        "intent": intent,  # type: ignore[typeddict-item]
        "intent_confidence": float(data.get("intent_confidence") or 0.0),
        "needs_clarification": bool(data.get("needs_clarification")) or intent == "unclear",
        "should_escalate": bool(data.get("should_escalate")) or intent == "escalate",
        "patient_id": data.get("patient_hint") or state.get("patient_id"),
        "error": "",
    }


def resolve_identity(state: EmailState) -> EmailState:
    """Map sender → patient_id. Stub: keep classifier hint or use from_address."""
    patient_id = state.get("patient_id") or state["from_address"]
    return {**state, "patient_id": patient_id}


def fetch_records(state: EmailState) -> EmailState:
    result = fetch_history_records(
        patient_id=state.get("patient_id"),
        from_address=state["from_address"],
    )
    return {
        **state,
        "tool_name": "fetch_history_records",
        "tool_result": result,
    }


def draft_reply(state: EmailState) -> EmailState:
    if state.get("should_escalate"):
        return {
            **state,
            "reply_subject": f"Re: {state['subject']}",
            "reply_body": (
                "Thank you for your email. A team member will follow up with you shortly."
            ),
            "tool_name": state.get("tool_name"),
            "tool_result": state.get("tool_result"),
        }

    if state.get("needs_clarification") and state.get("intent") == "unclear":
        return {
            **state,
            "reply_subject": f"Re: {state['subject']}",
            "reply_body": (
                "Thanks for reaching out. Could you please clarify what you need "
                "(for example: visit history, appointment, or billing) and confirm "
                "the email on your patient file?"
            ),
        }

    raw = chat_completion(
        [
            {"role": "system", "content": DRAFT_SYSTEM},
            {
                "role": "user",
                "content": build_draft_prompt(
                    subject=state["subject"],
                    body_text=state["body_text"],
                    intent=str(state.get("intent") or ""),
                    tool_result=state.get("tool_result"),
                ),
            },
        ]
    )
    data = _parse_json(raw)
    return {
        **state,
        "reply_subject": str(data.get("reply_subject") or f"Re: {state['subject']}"),
        "reply_body": str(data.get("reply_body") or "Thank you for your email."),
    }


def route_after_classify(
    state: EmailState,
) -> Literal["resolve_identity", "draft_reply"]:
    if state.get("should_escalate") or state.get("intent") in {"unclear", "escalate"}:
        return "draft_reply"
    return "resolve_identity"


def route_after_identity(
    state: EmailState,
) -> Literal["fetch_records", "draft_reply"]:
    if state.get("intent") == "history_records":
        return "fetch_records"
    return "draft_reply"


def build_email_graph():
    g = StateGraph(EmailState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("resolve_identity", resolve_identity)
    g.add_node("fetch_records", fetch_records)
    g.add_node("draft_reply", draft_reply)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"resolve_identity": "resolve_identity", "draft_reply": "draft_reply"},
    )
    g.add_conditional_edges(
        "resolve_identity",
        route_after_identity,
        {"fetch_records": "fetch_records", "draft_reply": "draft_reply"},
    )
    g.add_edge("fetch_records", "draft_reply")
    g.add_edge("draft_reply", END)
    return g.compile()


_GRAPH = None


def get_email_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_email_graph()
    return _GRAPH


def run_email_turn(
    *,
    message_id: str,
    from_address: str,
    subject: str,
    body_text: str,
    patient_id: str | None = None,
) -> EmailState:
    graph = get_email_graph()
    initial: EmailState = {
        "message_id": message_id,
        "from_address": from_address,
        "subject": subject,
        "body_text": body_text,
        "intent": None,
        "intent_confidence": 0.0,
        "patient_id": patient_id,
        "tool_name": None,
        "tool_result": None,
        "reply_subject": "",
        "reply_body": "",
        "needs_clarification": False,
        "should_escalate": False,
        "error": "",
    }
    return graph.invoke(initial)  # type: ignore[return-value]
