"""Shared continuity helpers for voice and email agents."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.db import continuity_repo as crepo
from app.db.models import IntakeProfile, User
from app.logging_setup import get_logger

log = get_logger(__name__)

VERIFIED = "verified"
UNVERIFIED = "unverified"
FAILED = "failed"


def is_verified(user: User | None) -> bool:
    return bool(user and user.verification_status == VERIFIED)


def try_verify_identity(
    *,
    profile: IntakeProfile | None,
    caller_name: str | None,
    child_dob: str | None,
    contact_match: bool,
) -> bool:
    """
    Phase-1 verification: caller name + child DOB must match an existing profile,
    and the inbound channel contact (phone/email) must already be linked.
    New users with no profile yet are not 'verified for disclosure' until a
    profile exists from a prior interaction — first contact starts fresh.
    """
    if not contact_match:
        return False
    if profile is None:
        return False
    if not profile.caller_name or not profile.child_dob:
        return False
    if not caller_name or not child_dob:
        return False
    name_ok = _norm_name(caller_name) == _norm_name(profile.caller_name)
    dob_ok = _norm_dob(child_dob) == _norm_dob(profile.child_dob)
    return name_ok and dob_ok


def _norm_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _norm_dob(value: str) -> str:
    # Keep digits only so "March 14, 2024" vs "2024-03-14" still needs LLM
    # normalization into ISO before verify; compare loose digit strings as fallback.
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits or value.strip().lower()


def build_continuity_block(
    db: Session,
    user_id: str | None,
    *,
    verified: bool,
    interaction_limit: int = 5,
) -> str:
    """
    Prompt block for Ava. Unverified: only a non-disclosing hint.
    Verified: intake snapshot + recent channel-neutral summaries.
    """
    if not user_id:
        return "CONTINUITY: no linked user yet."

    user = crepo.get_user(db, user_id)
    if user is None:
        return "CONTINUITY: no linked user yet."

    interactions = crepo.get_recent_interactions(db, user_id, limit=interaction_limit)
    has_history = bool(interactions) or crepo.get_intake_profile(db, user_id) is not None

    if not verified:
        if has_history:
            return (
                "CONTINUITY: A possible prior contact record may exist for this "
                "phone/email. Do NOT disclose diagnosis, schedule, coverage, "
                "member IDs, balances, or prior clinical details until identity "
                "and guardian authority are verified (caller name + child DOB + "
                "contact match). Ask only for verification fields. If verification "
                "fails or authority is contested, create a Tier 1 handoff without "
                "confirming whether the named person is a patient."
            )
        return "CONTINUITY: No prior interactions on file for this contact."

    profile = crepo.get_intake_profile(db, user_id)
    profile_dict = crepo.profile_to_dict(profile)
    lines = [
        "CONTINUITY (verified — minimum necessary context):",
        f"- user_id: {user_id}",
        f"- verification_status: {user.verification_status}",
        "- intake_profile:",
        json.dumps(_public_profile_snapshot(profile_dict), indent=2),
        "- recent_interaction_summaries (newest first):",
    ]
    if not interactions:
        lines.append("  (none)")
    else:
        for item in interactions:
            lines.append(
                f"  - [{item.channel}] outcome={item.outcome!r} "
                f"at={item.created_at}: {item.summary}"
            )
    lines.append(
        "Continue from the next unresolved outcome. Restate only minimum "
        "context; do not dump a prior transcript."
    )
    return "\n".join(lines)


def _public_profile_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    """Drop empty keys for a tighter prompt."""
    out: dict[str, Any] = {}
    for key, value in profile.items():
        if key == "capture":
            if isinstance(value, dict) and value:
                out[key] = value
            continue
        if value is not None and value != "":
            out[key] = value
    return out


def load_continuity_context(
    db: Session,
    user_id: str | None,
) -> dict[str, Any]:
    """Structured continuity payload for tools / state."""
    if not user_id:
        return {
            "user_id": None,
            "verified": False,
            "has_history": False,
            "profile": {},
            "interactions": [],
            "prompt_block": build_continuity_block(db, None, verified=False),
        }
    user = crepo.get_user(db, user_id)
    verified = is_verified(user)
    profile = crepo.get_intake_profile(db, user_id)
    interactions = crepo.get_recent_interactions(db, user_id, limit=5)
    return {
        "user_id": user_id,
        "verified": verified,
        "has_history": bool(interactions) or profile is not None,
        "profile": crepo.profile_to_dict(profile),
        "interactions": [
            {
                "channel": i.channel,
                "summary": i.summary,
                "outcome": i.outcome,
                "external_id": i.external_id,
                "created_at": i.created_at.isoformat() if i.created_at else None,
            }
            for i in interactions
        ],
        "prompt_block": build_continuity_block(db, user_id, verified=verified),
    }


def persist_channel_turn(
    db: Session,
    *,
    user_id: str,
    channel: str,
    external_id: str | None,
    summary: str,
    outcome: str | None,
    verification_status: str,
    profile_fields: dict[str, Any] | None = None,
    capture_delta: dict[str, Any] | None = None,
) -> Interaction:
    """Upsert intake profile fields and append a channel-neutral interaction."""
    fields = dict(profile_fields or {})
    if capture_delta:
        fields["capture_delta"] = capture_delta
    # Split known columns vs capture blob
    known_keys = {
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
        "primary_disposition",
        "intake_complete",
    }
    upsert_kwargs: dict[str, Any] = {
        k: fields[k] for k in known_keys if k in fields and fields[k] is not None
    }
    delta = fields.get("capture_delta") if isinstance(fields.get("capture_delta"), dict) else {}
    # Also fold non-known keys into capture
    extra = {k: v for k, v in fields.items() if k not in known_keys and k != "capture_delta"}
    merged_delta = {**extra, **(delta or {})}
    if upsert_kwargs or merged_delta:
        crepo.upsert_intake_profile(
            db,
            user_id,
            capture_delta=merged_delta or None,
            **upsert_kwargs,
        )
    return crepo.append_interaction(
        db,
        user_id=user_id,
        channel=channel,
        summary=summary,
        external_id=external_id,
        outcome=outcome,
        verification_status=verification_status,
        capture_delta=merged_delta or None,
    )


def summarize_interaction_from_capture(
    *,
    channel: str,
    disposition: str | None,
    capture: dict[str, Any],
    reply_excerpt: str | None = None,
) -> str:
    """Build a short channel-neutral summary for the interactions table."""
    parts: list[str] = [f"{channel} interaction"]
    child = capture.get("child_first_name")
    if child:
        parts.append(f"child={child}")
    dx = capture.get("diagnosis_stated")
    if dx:
        parts.append(f"dx_stated={dx}")
    carrier = capture.get("insurance_carrier")
    if carrier:
        parts.append(f"coverage={carrier}")
    if disposition:
        parts.append(f"disposition={disposition}")
    open_items = capture.get("secondary_tasks") or capture.get("next_steps")
    if open_items:
        parts.append(f"open={open_items!r}")
    if reply_excerpt:
        parts.append(f"close_note={reply_excerpt[:160]}")
    return "; ".join(parts)
