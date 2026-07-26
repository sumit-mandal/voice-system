"""LangGraph state for healthcare phone intake."""

from __future__ import annotations

from typing import Literal, TypedDict

PendingField = Literal["name", "age", "ready", "diseases", "medications"]


class IntakeState(TypedDict):
    call_sid: str
    user_text: str
    messages: list[dict[str, str]]
    patient_name: str | None
    patient_age: int | None
    ready_to_proceed: bool | None
    diseases: str | None
    medications: str | None
    pending_field: PendingField | None
    reply: str
    is_complete: bool
    should_end: bool
    validation_notes: str
