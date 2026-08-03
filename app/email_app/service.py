"""Orchestrate: inbound email fields → LangGraph → SES auto-reply."""

from __future__ import annotations

import uuid
from typing import Any

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

    return {
        "message_id": mid,
        "from_address": from_addr,
        "intent": result.get("intent"),
        "intent_confidence": result.get("intent_confidence"),
        "patient_id": result.get("patient_id"),
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
