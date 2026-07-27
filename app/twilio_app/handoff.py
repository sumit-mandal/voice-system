"""Cold-transfer a live Twilio call from the AI agent to a human number."""

from __future__ import annotations

import re

from twilio.rest import Client

from app.config import get_settings
from app.logging_setup import get_logger

log = get_logger(__name__)

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")


def is_twilio_call_sid(call_sid: str) -> bool:
    """Real Twilio CallSids look like CA…; browser/debug sessions do not."""
    return bool(call_sid) and call_sid.startswith("CA") and len(call_sid) >= 34


def redirect_call_to_human(*, call_sid: str) -> None:
    """
    Point the live Twilio call at /twilio/voice/handoff so Twilio dials the human.

    Raises ValueError if credentials / agent number are missing or CallSid is not Twilio.
    """
    if not is_twilio_call_sid(call_sid):
        raise ValueError(f"Not a Twilio CallSid — cannot redirect | call_sid={call_sid}")

    settings = get_settings()
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        raise ValueError("Twilio credentials not configured")
    agent = (settings.twilio_human_agent_number or "").strip()
    if not _E164.match(agent):
        raise ValueError(
            "TWILIO_HUMAN_AGENT_NUMBER must be E.164, e.g. +9198XXXXXXXX"
        )

    base = settings.public_base_url.strip().rstrip("/")
    handoff_url = f"{base}/twilio/voice/handoff"
    log.info(
        "Redirecting Twilio call to human | CallSid=%s url=%s agent=%s",
        call_sid,
        handoff_url,
        agent,
    )
    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    call = client.calls(call_sid).update(url=handoff_url, method="POST")
    log.info(
        "Twilio call redirect queued | CallSid=%s status=%s",
        call.sid,
        call.status,
    )
