"""Email LangGraph: classify → identity → continuity → draft reply."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.agent.llm import chat_completion
from app.continuity.context import (
    build_continuity_block,
    is_verified,
    try_verify_identity,
)
from app.db import continuity_repo as crepo
from app.db.clinic_repo import get_clinic_name
from app.db.session import SessionLocal
from app.email_agent.prompts import (
    build_classify_prompt,
    build_classify_system,
    build_draft_prompt,
    build_draft_system,
)
from app.email_agent.state import EmailState
from app.email_agent.tools.records import fetch_history_records
from app.logging_setup import get_logger

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_VALID_INTENTS: set[str] = {
    "new_intake",
    "continue_intake",
    "records_request",
    "benefits_status",
    "human_handoff",
    "unclear",
    "escalate",
}

_FETCH_INTENTS = {
    "continue_intake",
    "records_request",
    "benefits_status",
    "new_intake",
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
            {"role": "system", "content": build_classify_system()},
            {
                "role": "user",
                "content": build_classify_prompt(
                    subject=state["subject"],
                    body_text=state["body_text"],
                    from_address=state["from_address"],
                    continuity_block=state.get("continuity_block") or "",
                ),
            },
        ]
    )
    data = _parse_json(raw)
    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        intent = "unclear"

    capture = dict(state.get("capture") or {})
    for key in (
        "caller_name",
        "child_first_name",
        "child_dob",
        "diagnosis_stated",
        "insurance_carrier",
    ):
        if data.get(key):
            capture[key] = data[key]

    return {
        **state,
        "intent": intent,  # type: ignore[typeddict-item]
        "intent_confidence": float(data.get("intent_confidence") or 0.0),
        "needs_clarification": bool(data.get("needs_clarification")) or intent == "unclear",
        "should_escalate": bool(data.get("should_escalate"))
        or intent in {"escalate", "human_handoff"},
        "capture": capture,
        "error": "",
    }


def resolve_identity(state: EmailState) -> EmailState:
    """Map sender email → user UUID; gate disclosure via verification."""
    db = SessionLocal()
    try:
        user = crepo.find_or_create_user_by_email(db, state["from_address"])
        if user is None:
            return {
                **state,
                "user_id": None,
                "patient_id": None,
                "identity_verified": False,
                "continuity_block": "CONTINUITY: no linked user yet.",
            }

        user_id = user.id
        profile = crepo.get_intake_profile(db, user_id)
        capture = state.get("capture") or {}
        verified = is_verified(user)

        matched = crepo.find_user_by_caller_and_child_dob(
            db,
            caller_name=capture.get("caller_name"),
            child_dob=capture.get("child_dob"),
        )
        if matched and matched.id != user_id:
            crepo.merge_channel_onto_user(
                db,
                from_user_id=user_id,
                to_user_id=matched.id,
                channel="email",
                value=state["from_address"],
            )
            user_id = matched.id
            profile = crepo.get_intake_profile(db, user_id)
            crepo.set_verification_status(
                db, user_id, status="verified", method="email_name_dob_cross_channel"
            )
            verified = True
        elif not verified:
            if try_verify_identity(
                profile=profile,
                caller_name=capture.get("caller_name"),
                child_dob=capture.get("child_dob"),
                contact_match=True,
            ):
                crepo.set_verification_status(
                    db, user_id, status="verified", method="email_name_dob"
                )
                verified = True

        block = build_continuity_block(db, user_id, verified=verified)
        log.info(
            "resolve_identity | user_id=%s verified=%s from=%s",
            user_id,
            verified,
            state["from_address"],
        )
        return {
            **state,
            "user_id": user_id,
            "patient_id": user_id,
            "identity_verified": verified,
            "continuity_block": block,
        }
    finally:
        db.close()


def fetch_records(state: EmailState) -> EmailState:
    result = fetch_history_records(
        patient_id=state.get("user_id") or state.get("patient_id"),
        from_address=state["from_address"],
    )
    return {
        **state,
        "tool_name": "fetch_history_records",
        "tool_result": result,
        "continuity_block": result.get("prompt_block")
        or state.get("continuity_block")
        or "",
        "identity_verified": bool(result.get("verified"))
        if "verified" in result
        else state.get("identity_verified", False),
    }


def draft_reply(state: EmailState) -> EmailState:
    clinic = get_clinic_name()
    if state.get("should_escalate") or state.get("intent") in {
        "escalate",
        "human_handoff",
    }:
        return {
            **state,
            "reply_subject": f"Re: {state['subject']}" if state["subject"] else "We will call you",
            "reply_body": (
                f"Thank you for reaching out to {clinic}. A team member will follow "
                "up with you shortly by phone.\n\n"
                "This email may contain confidential information intended only for the "
                "addressee. If you received it in error, please delete it and notify us. "
                "Do not include diagnoses, member IDs, or clinical details in email subject lines."
            ),
            "primary_disposition": "Warm Transfer",
            "interaction_summary": "Email escalated to human handoff.",
            "capture": {
                **(state.get("capture") or {}),
                "primary_disposition": "Warm Transfer",
            },
        }

    if state.get("needs_clarification") and state.get("intent") == "unclear":
        return {
            **state,
            "reply_subject": f"Quick clarification — {clinic}",
            "reply_body": (
                "Thanks for writing. To help the right way, could you reply with:\n"
                "1) Your name and relationship to the child\n"
                "2) The child's first name and date of birth\n"
                "3) Whether you need intake, records upload, or benefits help\n\n"
                "This email may contain confidential information intended only for the "
                "addressee. If you received it in error, please delete it and notify us. "
                "Do not include diagnoses, member IDs, or clinical details in email subject lines."
            ),
            "interaction_summary": "Email unclear; asked for clarification fields.",
        }

    raw = chat_completion(
        [
            {"role": "system", "content": build_draft_system(clinic)},
            {
                "role": "user",
                "content": build_draft_prompt(
                    subject=state["subject"],
                    body_text=state["body_text"],
                    intent=str(state.get("intent") or ""),
                    tool_result=state.get("tool_result"),
                    continuity_block=state.get("continuity_block") or "",
                    identity_verified=bool(state.get("identity_verified")),
                ),
            },
        ]
    )
    data = _parse_json(raw)
    capture = dict(state.get("capture") or {})
    if isinstance(data.get("capture"), dict):
        capture.update({k: v for k, v in data["capture"].items() if v is not None})
    disposition = data.get("primary_disposition")
    if disposition:
        capture["primary_disposition"] = disposition

    return {
        **state,
        "reply_subject": str(data.get("reply_subject") or f"Re: {state['subject']}"),
        "reply_body": str(data.get("reply_body") or "Thank you for your email."),
        "primary_disposition": disposition,
        "capture": capture,
        "interaction_summary": str(
            data.get("interaction_summary")
            or f"Email {state.get('intent')} reply drafted."
        ),
        "identity_verified": bool(data.get("identity_verified"))
        if data.get("identity_verified") is not None
        else state.get("identity_verified", False),
    }


def route_after_classify(
    state: EmailState,
) -> Literal["resolve_identity", "draft_reply"]:
    if state.get("intent") in {"unclear"} and state.get("needs_clarification"):
        # Still resolve identity so we can persist the thread to a user
        return "resolve_identity"
    return "resolve_identity"


def route_after_identity(
    state: EmailState,
) -> Literal["fetch_records", "draft_reply"]:
    if state.get("should_escalate"):
        return "draft_reply"
    if state.get("intent") in _FETCH_INTENTS:
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
    continuity_block = "CONTINUITY: no linked user yet."
    identity_verified = False
    user_id = patient_id

    if user_id:
        db = SessionLocal()
        try:
            user = crepo.get_user(db, user_id)
            identity_verified = is_verified(user)
            continuity_block = build_continuity_block(
                db, user_id, verified=identity_verified
            )
        finally:
            db.close()

    initial: EmailState = {
        "message_id": message_id,
        "from_address": from_address,
        "subject": subject,
        "body_text": body_text,
        "intent": None,
        "intent_confidence": 0.0,
        "user_id": user_id,
        "patient_id": user_id,
        "identity_verified": identity_verified,
        "continuity_block": continuity_block,
        "tool_name": None,
        "tool_result": None,
        "capture": {},
        "primary_disposition": None,
        "interaction_summary": "",
        "reply_subject": "",
        "reply_body": "",
        "needs_clarification": False,
        "should_escalate": False,
        "error": "",
    }
    return graph.invoke(initial)  # type: ignore[return-value]
