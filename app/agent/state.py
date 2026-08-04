"""LangGraph state for Ava phone intake."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

PendingField = Literal[
    "recording_notice",
    "caller_name",
    "relationship",
    "child_name",
    "child_dob",
    "verify_identity",
    "diagnosis",
    "insurance",
    "location",
    "callback",
    "services",
    "consent",
    "close",
    None,
]

PrimaryDisposition = Literal[
    "Scheduled",
    "Warm Transfer",
    "Callback Queue",
    "Records Needed",
    "Referred Out",
    "Waitlist",
    "Safety Stop",
    "Verification Failed",
]


class IntakeState(TypedDict):
    call_sid: str
    user_id: str | None
    user_text: str
    messages: list[dict[str, str]]
    # Continuity
    identity_verified: bool
    continuity_block: str
    recording_notice_delivered: bool
    # Ava capture (critical typed slots)
    caller_name: str | None
    relationship_to_child: str | None
    callback_number: str | None
    child_first_name: str | None
    child_last_name: str | None
    child_dob: str | None
    child_age_computed: str | None
    home_city: str | None
    home_zip: str | None
    diagnosis_stated: str | None
    asd_diagnosis: bool | None
    insurance_carrier: str | None
    primary_disposition: str | None
    intake_complete: bool | None
    capture: dict[str, Any]
    pending_field: PendingField | None
    # Dialogue control
    reply: str
    is_complete: bool
    should_end: bool
    handoff_requested: bool
    handoff_reason: str
    handoff_summary: str
    validation_notes: str
    # Legacy compatibility fields still read by some DB/debug paths
    patient_name: str | None
    patient_age: int | None
    ready_to_proceed: bool | None
    diseases: str | None
    medications: str | None
