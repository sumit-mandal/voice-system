"""Orchestrate: inbound email fields → LangGraph → SES auto-reply → persist continuity."""

from __future__ import annotations

import uuid
from typing import Any

from app.continuity.context import persist_channel_turn, summarize_interaction_from_capture
from app.db import continuity_repo as crepo
from app.db.session import SessionLocal
from app.email_agent.graph import run_email_turn
from app.email_app.ses_client import normalize_address, send_auto_reply
from app.logging_setup import get_logger

log = get_logger(__name__)


def process_and_reply(
    *,
    from_address: str,
    subject: str,
    body_text: str,
    message_id: str | None = None,
    send_reply: bool = True,
    patient_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the email agent and optionally send the drafted reply via SES.

    In SES sandbox, `from_address` (reply destination) must be a verified identity.
    """
    mid = message_id or f"local-{uuid.uuid4().hex[:12]}"
    from_addr = normalize_address(from_address)
    log.info(
        "email process_and_reply | from=%s subject=%r send_reply=%s message_id=%s",
        from_addr,
        subject,
        send_reply,
        mid,
    )

    result = run_email_turn(
        message_id=mid,
        from_address=from_addr,
        subject=subject or "(no subject)",
        body_text=body_text or "",
        patient_id=patient_id,
    )

    ses_message_id: str | None = None
    send_error: str | None = None
    if send_reply:
        try:
            ses_message_id = send_auto_reply(
                to_address=from_addr,
                subject=result["reply_subject"] or f"Re: {subject}",
                body_text=result["reply_body"] or "",
                in_reply_to=mid if mid.startswith("<") else f"<{mid}@local>",
            )
        except Exception as exc:
            send_error = str(exc)
            log.exception("Auto-reply send failed")

    user_id = result.get("user_id") or result.get("patient_id")
    if user_id:
        _persist_email_interaction(
            user_id=user_id,
            message_id=mid,
            result=result,
            from_address=from_addr,
        )

    return {
        "message_id": mid,
        "from_address": from_addr,
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),
        "patient_id": result.get("patient_id") or user_id,
        "user_id": user_id,
        "identity_verified": result.get("identity_verified"),
        "primary_disposition": result.get("primary_disposition"),
        "tool_name": result.get("tool_name"),
        "tool_result": result.get("tool_result"),
        "reply_subject": result.get("reply_subject"),
        "reply_body": result.get("reply_body"),
        "needs_clarification": result.get("needs_clarification"),
        "should_escalate": result.get("should_escalate"),
        "ses_message_id": ses_message_id,
        "send_error": send_error,
        "sent": bool(ses_message_id),
    }


def _persist_email_interaction(
    *,
    user_id: str,
    message_id: str,
    result: dict[str, Any],
    from_address: str,
) -> None:
    capture = dict(result.get("capture") or {})
    disposition = result.get("primary_disposition") or capture.get("primary_disposition")
    profile_fields = {
        k: capture.get(k)
        for k in (
            "caller_name",
            "relationship_to_child",
            "callback_number",
            "child_first_name",
            "child_last_name",
            "child_dob",
            "child_age_computed",
            "home_city",
            "home_zip",
            "diagnosis_stated",
            "asd_diagnosis",
            "insurance_carrier",
            "intake_complete",
        )
        if capture.get(k) is not None
    }
    if disposition:
        profile_fields["primary_disposition"] = disposition

    summary = result.get("interaction_summary") or summarize_interaction_from_capture(
        channel="email",
        disposition=disposition,
        capture=capture,
        reply_excerpt=result.get("reply_body"),
    )

    db = SessionLocal()
    try:
        persist_channel_turn(
            db,
            user_id=user_id,
            channel="email",
            external_id=message_id,
            summary=summary,
            outcome=disposition or str(result.get("intent") or "email"),
            verification_status=(
                "verified" if result.get("identity_verified") else "unverified"
            ),
            profile_fields=profile_fields,
            capture_delta=capture,
        )
        # Ensure email identity stays linked
        crepo.link_channel_identity(
            db, user_id=user_id, channel="email", value=from_address
        )
    finally:
        db.close()
