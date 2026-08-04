"""Repository helpers for shared user continuity across voice and email."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import ChannelIdentity, IntakeProfile, Interaction, User
from app.logging_setup import get_logger

log = get_logger(__name__)

_NON_DIGIT = re.compile(r"\D+")


def normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    if not raw or raw.lower() in {"browser", "debug"}:
        return None
    digits = _NON_DIGIT.sub("", raw)
    if not digits:
        return None
    if raw.startswith("+") and digits:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}" if not raw.startswith("+") else f"+{digits}"


def normalize_email(value: str | None) -> str | None:
    if not value:
        return None
    addr = value.strip().lower()
    # Strip display-name wrappers: "Name <a@b.com>"
    if "<" in addr and ">" in addr:
        start = addr.rfind("<") + 1
        end = addr.rfind(">")
        addr = addr[start:end].strip()
    return addr or None


def _new_user_id() -> str:
    return str(uuid.uuid4())


def get_user(db: Session, user_id: str) -> User | None:
    return db.query(User).filter(User.id == user_id).one_or_none()


def find_user_by_phone(db: Session, phone: str | None) -> User | None:
    norm = normalize_phone(phone)
    if not norm:
        return None
    ident = (
        db.query(ChannelIdentity)
        .filter(ChannelIdentity.channel == "phone", ChannelIdentity.value == norm)
        .one_or_none()
    )
    if ident is None:
        return None
    return get_user(db, ident.user_id)


def find_user_by_email(db: Session, email: str | None) -> User | None:
    norm = normalize_email(email)
    if not norm:
        return None
    ident = (
        db.query(ChannelIdentity)
        .filter(ChannelIdentity.channel == "email", ChannelIdentity.value == norm)
        .one_or_none()
    )
    if ident is None:
        return None
    return get_user(db, ident.user_id)


def find_or_create_user_by_phone(db: Session, phone: str | None) -> User | None:
    norm = normalize_phone(phone)
    if not norm:
        log.debug("find_or_create_user_by_phone | unusable phone=%r", phone)
        return None
    existing = find_user_by_phone(db, norm)
    if existing is not None:
        log.debug("Found user by phone | user_id=%s phone=%s", existing.id, norm)
        return existing
    user = User(id=_new_user_id(), verification_status="unverified")
    db.add(user)
    db.flush()
    db.add(ChannelIdentity(user_id=user.id, channel="phone", value=norm))
    db.commit()
    db.refresh(user)
    log.info("Created user by phone | user_id=%s phone=%s", user.id, norm)
    return user


def find_or_create_user_by_email(db: Session, email: str | None) -> User | None:
    norm = normalize_email(email)
    if not norm:
        log.debug("find_or_create_user_by_email | unusable email=%r", email)
        return None
    existing = find_user_by_email(db, norm)
    if existing is not None:
        log.debug("Found user by email | user_id=%s email=%s", existing.id, norm)
        return existing
    user = User(id=_new_user_id(), verification_status="unverified")
    db.add(user)
    db.flush()
    db.add(ChannelIdentity(user_id=user.id, channel="email", value=norm))
    db.commit()
    db.refresh(user)
    log.info("Created user by email | user_id=%s email=%s", user.id, norm)
    return user


def find_user_by_caller_and_child_dob(
    db: Session,
    *,
    caller_name: str | None,
    child_dob: str | None,
) -> User | None:
    """Find an existing user whose intake profile matches caller name + child DOB."""
    if not caller_name or not child_dob:
        return None
    target_name = " ".join(caller_name.strip().lower().split())
    target_dob_digits = "".join(ch for ch in child_dob if ch.isdigit())
    profiles = db.query(IntakeProfile).all()
    for profile in profiles:
        if not profile.caller_name or not profile.child_dob:
            continue
        name_ok = " ".join(profile.caller_name.strip().lower().split()) == target_name
        dob_digits = "".join(ch for ch in profile.child_dob if ch.isdigit())
        dob_ok = bool(target_dob_digits) and dob_digits == target_dob_digits
        if name_ok and dob_ok:
            return get_user(db, profile.user_id)
    return None


def merge_channel_onto_user(
    db: Session,
    *,
    from_user_id: str,
    to_user_id: str,
    channel: str,
    value: str,
) -> User:
    """
    Point a channel identity at to_user_id. Used when name+DOB match reveals
    the phone/email contact belongs to an existing family UUID.
    """
    if from_user_id == to_user_id:
        link_channel_identity(db, user_id=to_user_id, channel=channel, value=value)
        return get_user(db, to_user_id)  # type: ignore[return-value]

    if channel == "phone":
        norm = normalize_phone(value)
    else:
        norm = normalize_email(value)
    if not norm:
        return get_user(db, to_user_id)  # type: ignore[return-value]

    ident = (
        db.query(ChannelIdentity)
        .filter(ChannelIdentity.channel == channel, ChannelIdentity.value == norm)
        .one_or_none()
    )
    if ident is not None:
        ident.user_id = to_user_id
    else:
        db.add(ChannelIdentity(user_id=to_user_id, channel=channel, value=norm))
    db.commit()
    log.info(
        "Merged channel onto user | from=%s to=%s channel=%s value=%s",
        from_user_id,
        to_user_id,
        channel,
        norm,
    )
    return get_user(db, to_user_id)  # type: ignore[return-value]


def link_channel_identity(
    db: Session,
    *,
    user_id: str,
    channel: str,
    value: str,
) -> ChannelIdentity | None:
    """Attach another channel identity to an existing user (e.g. phone after email)."""
    if channel == "phone":
        norm = normalize_phone(value)
    elif channel == "email":
        norm = normalize_email(value)
    else:
        raise ValueError(f"unsupported channel={channel}")
    if not norm:
        return None

    existing = (
        db.query(ChannelIdentity)
        .filter(ChannelIdentity.channel == channel, ChannelIdentity.value == norm)
        .one_or_none()
    )
    if existing is not None:
        if existing.user_id != user_id:
            log.warning(
                "Channel identity already linked to another user | channel=%s value=%s "
                "existing_user=%s requested=%s",
                channel,
                norm,
                existing.user_id,
                user_id,
            )
        return existing

    row = ChannelIdentity(user_id=user_id, channel=channel, value=norm)
    db.add(row)
    db.commit()
    db.refresh(row)
    log.info(
        "Linked channel identity | user_id=%s channel=%s value=%s",
        user_id,
        channel,
        norm,
    )
    return row


def set_verification_status(
    db: Session,
    user_id: str,
    *,
    status: str,
    method: str | None = None,
) -> User:
    user = get_user(db, user_id)
    if user is None:
        raise ValueError(f"No user for user_id={user_id}")
    user.verification_status = status
    user.verification_method = method
    if status == "verified":
        user.verified_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    log.info(
        "Set verification | user_id=%s status=%s method=%s",
        user_id,
        status,
        method,
    )
    return user


def get_intake_profile(db: Session, user_id: str) -> IntakeProfile | None:
    return db.query(IntakeProfile).filter(IntakeProfile.user_id == user_id).one_or_none()


def _merge_capture(existing_json: str | None, delta: dict[str, Any] | None) -> str | None:
    base: dict[str, Any] = {}
    if existing_json:
        try:
            loaded = json.loads(existing_json)
            if isinstance(loaded, dict):
                base = loaded
        except json.JSONDecodeError:
            base = {}
    if delta:
        for key, value in delta.items():
            if value is not None:
                base[key] = value
    return json.dumps(base) if base else existing_json


def upsert_intake_profile(
    db: Session,
    user_id: str,
    *,
    caller_name: str | None = None,
    relationship_to_child: str | None = None,
    callback_number: str | None = None,
    child_first_name: str | None = None,
    child_last_name: str | None = None,
    child_dob: str | None = None,
    child_age_computed: str | None = None,
    home_city: str | None = None,
    home_zip: str | None = None,
    diagnosis_stated: str | None = None,
    asd_diagnosis: bool | None = None,
    insurance_carrier: str | None = None,
    primary_disposition: str | None = None,
    intake_complete: bool | None = None,
    capture_delta: dict[str, Any] | None = None,
) -> IntakeProfile:
    row = get_intake_profile(db, user_id)
    if row is None:
        row = IntakeProfile(user_id=user_id)
        db.add(row)

    if caller_name is not None:
        row.caller_name = caller_name
    if relationship_to_child is not None:
        row.relationship_to_child = relationship_to_child
    if callback_number is not None:
        row.callback_number = callback_number
    if child_first_name is not None:
        row.child_first_name = child_first_name
    if child_last_name is not None:
        row.child_last_name = child_last_name
    if child_dob is not None:
        row.child_dob = child_dob
    if child_age_computed is not None:
        row.child_age_computed = child_age_computed
    if home_city is not None:
        row.home_city = home_city
    if home_zip is not None:
        row.home_zip = home_zip
    if diagnosis_stated is not None:
        row.diagnosis_stated = diagnosis_stated
    if asd_diagnosis is not None:
        row.asd_diagnosis = asd_diagnosis
    if insurance_carrier is not None:
        row.insurance_carrier = insurance_carrier
    if primary_disposition is not None:
        row.primary_disposition = primary_disposition
    if intake_complete is not None:
        row.intake_complete = intake_complete
    if capture_delta:
        row.capture_json = _merge_capture(row.capture_json, capture_delta)

    db.commit()
    db.refresh(row)
    log.info(
        "Upserted intake_profile | user_id=%s disposition=%s child=%s",
        user_id,
        row.primary_disposition,
        row.child_first_name,
    )
    return row


def append_interaction(
    db: Session,
    *,
    user_id: str,
    channel: str,
    summary: str,
    external_id: str | None = None,
    outcome: str | None = None,
    verification_status: str = "unverified",
    capture_delta: dict[str, Any] | None = None,
) -> Interaction:
    row = Interaction(
        user_id=user_id,
        channel=channel,
        external_id=external_id,
        summary=summary.strip(),
        outcome=outcome,
        verification_status=verification_status,
        capture_delta_json=json.dumps(capture_delta) if capture_delta else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    log.info(
        "Appended interaction | id=%s user_id=%s channel=%s outcome=%s",
        row.id,
        user_id,
        channel,
        outcome,
    )
    return row


def get_recent_interactions(
    db: Session,
    user_id: str,
    *,
    limit: int = 5,
) -> list[Interaction]:
    return (
        db.query(Interaction)
        .filter(Interaction.user_id == user_id)
        .order_by(Interaction.created_at.desc())
        .limit(limit)
        .all()
    )


def profile_to_dict(profile: IntakeProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    capture: dict[str, Any] = {}
    if profile.capture_json:
        try:
            loaded = json.loads(profile.capture_json)
            if isinstance(loaded, dict):
                capture = loaded
        except json.JSONDecodeError:
            capture = {}
    return {
        "caller_name": profile.caller_name,
        "relationship_to_child": profile.relationship_to_child,
        "callback_number": profile.callback_number,
        "child_first_name": profile.child_first_name,
        "child_last_name": profile.child_last_name,
        "child_dob": profile.child_dob,
        "child_age_computed": profile.child_age_computed,
        "home_city": profile.home_city,
        "home_zip": profile.home_zip,
        "diagnosis_stated": profile.diagnosis_stated,
        "asd_diagnosis": profile.asd_diagnosis,
        "insurance_carrier": profile.insurance_carrier,
        "primary_disposition": profile.primary_disposition,
        "intake_complete": profile.intake_complete,
        "capture": capture,
    }
