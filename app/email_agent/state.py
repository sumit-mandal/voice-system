"""State for Ava email LangGraph agent."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

EmailIntent = Literal[
    "new_intake",
    "continue_intake",
    "records_request",
    "benefits_status",
    "human_handoff",
    "unclear",
    "escalate",
]


class EmailState(TypedDict):
    message_id: str
    from_address: str
    subject: str
    body_text: str

    intent: EmailIntent | None
    intent_confidence: float
    user_id: str | None
    patient_id: str | None  # alias of user_id for API compatibility
    identity_verified: bool
    continuity_block: str

    tool_name: str | None
    tool_result: dict[str, Any] | None

    # Extracted / updated capture for persistence
    capture: dict[str, Any]
    primary_disposition: str | None
    interaction_summary: str

    reply_subject: str
    reply_body: str
    needs_clarification: bool
    should_escalate: bool
    error: str
