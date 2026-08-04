"""DB helpers for call sessions — loud debug, no silent swallow."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import CallSession
from app.logging_setup import get_logger

log = get_logger(__name__)


def create_call_session(
    db: Session,
    *,
    call_sid: str,
    room_name: str,
    caller_number: str | None,
    user_id: str | None = None,
) -> CallSession:
    log.debug(
        "DB create_call_session | call_sid=%s room=%s caller=%s user_id=%s",
        call_sid,
        room_name,
        caller_number,
        user_id,
    )
    row = CallSession(
        call_sid=call_sid,
        room_name=room_name,
        caller_number=caller_number,
        user_id=user_id,
        status="in_progress",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    log.info(
        "Created call_session id=%s call_sid=%s user_id=%s",
        row.id,
        row.call_sid,
        row.user_id,
    )
    return row


def set_call_user_id(db: Session, *, call_sid: str, user_id: str) -> CallSession:
    row = get_by_call_sid(db, call_sid)
    if row is None:
        raise ValueError(f"No call_session for call_sid={call_sid}")
    row.user_id = user_id
    db.commit()
    db.refresh(row)
    return row


def get_by_call_sid(db: Session, call_sid: str) -> CallSession | None:
    log.debug("DB get_by_call_sid | call_sid=%s", call_sid)
    return db.query(CallSession).filter(CallSession.call_sid == call_sid).one_or_none()


def get_by_room_name(db: Session, room_name: str) -> CallSession | None:
    log.debug("DB get_by_room_name | room=%s", room_name)
    return db.query(CallSession).filter(CallSession.room_name == room_name).one_or_none()


def update_intake(
    db: Session,
    *,
    call_sid: str,
    patient_name: str | None = None,
    patient_age: int | None = None,
    ready_to_proceed: bool | None = None,
    diseases: str | None = None,
    medications: str | None = None,
    transcript: str | None = None,
    status: str,
    handoff_reason: str | None = None,
    handoff_summary: str | None = None,
) -> CallSession:
    log.debug(
        "DB update_intake | call_sid=%s name=%s age=%s ready=%s diseases=%r meds=%r "
        "handoff_reason=%r status=%s",
        call_sid,
        patient_name,
        patient_age,
        ready_to_proceed,
        diseases,
        medications,
        handoff_reason,
        status,
    )
    row = get_by_call_sid(db, call_sid)
    if row is None:
        raise ValueError(f"No call_session for call_sid={call_sid}")

    if patient_name is not None:
        row.patient_name = patient_name
    if patient_age is not None:
        row.patient_age = patient_age
    if ready_to_proceed is not None:
        row.ready_to_proceed = ready_to_proceed
    if diseases is not None:
        row.diseases = diseases
    if medications is not None:
        row.medications = medications
    if transcript is not None:
        row.transcript = transcript
    if handoff_reason is not None:
        row.handoff_reason = handoff_reason
    if handoff_summary is not None:
        row.handoff_summary = handoff_summary
    row.status = status
    db.commit()
    db.refresh(row)
    log.info(
        "Updated call_session id=%s name=%s age=%s ready=%s status=%s",
        row.id,
        row.patient_name,
        row.patient_age,
        row.ready_to_proceed,
        row.status,
    )
    return row


def append_transcript(db: Session, *, call_sid: str, line: str) -> CallSession:
    log.debug("DB append_transcript | call_sid=%s line=%r", call_sid, line)
    row = get_by_call_sid(db, call_sid)
    if row is None:
        raise ValueError(f"No call_session for call_sid={call_sid}")
    existing = row.transcript or ""
    row.transcript = f"{existing}{line}\n"
    db.commit()
    db.refresh(row)
    return row
