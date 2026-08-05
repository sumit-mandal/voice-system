"""LangGraph Ava intake: extract/validate → save when complete, disposition, or handoff."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.agent.llm import chat_completion
from app.agent.prompts import build_system_prompt, build_user_prompt
from app.agent.state import IntakeState
from app.continuity.ava.policy import LIVE_FACTS
from app.continuity.context import (
    is_verified,
    persist_channel_turn,
    summarize_interaction_from_capture,
    try_verify_identity,
)
from app.continuity.context import build_continuity_block
from app.db import continuity_repo as crepo
from app.db import repository as repo
from app.db.session import SessionLocal
from app.logging_setup import get_logger

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_DEFAULT_HANDOFF_REPLY = (
    "Of course. I'll connect you with a specialist now. Please hold."
)
_DISPOSITIONS = {
    "Scheduled",
    "Warm Transfer",
    "Callback Queue",
    "Records Needed",
    "Referred Out",
    "Waitlist",
    "Safety Stop",
    "Verification Failed",
}
_MAX_UNCLEAR = 1  # after one unclear retry, skip field
_MAX_TURNS = 40  # emergency cap only; normal completion follows the full checklist
_FIELD_SKIP_ORDER = [
    "caller_name",
    "relationship",
    "callback",
    "best_callback_time",
    "child_name",
    "child_dob",
    "location",
    "language",
    "diagnosis",
    "diagnosing_provider",
    "insurance",
    "insurance_plan",
    "member_id",
    "services",
    "availability",
    "consent",
    "additional_notes",
    "close",
]

_FIELD_PROMPTS = {
    "caller_name": "Could I please get your first and last name?",
    "relationship": "What is your relationship to the child?",
    "callback": "What is the best callback number for you?",
    "best_callback_time": "What is the best time of day for our team to call you?",
    "child_name": "Could I get the child's first and last name?",
    "child_dob": "What is the child's date of birth?",
    "location": "What city and ZIP code does the child live in?",
    "language": "What language do you prefer for calls, and do you need an interpreter?",
    "diagnosis": "What diagnosis has the child received?",
    "diagnosing_provider": "Who provided the diagnosis, and about when was it made?",
    "insurance": "Who is the primary insurance carrier?",
    "insurance_plan": "What is the insurance plan or product name, if you know it?",
    "member_id": "What is the member ID and subscriber name?",
    "services": "Which services are you looking for, such as ABA, speech, or OT?",
    "availability": "What days or times and care setting work best for your family?",
    "consent": (
        "May we contact you at this number about intake, and what contact method is safest?"
    ),
    "additional_notes": "Before we finish, is there anything else you want the care team to know?",
}


def _parse_llm_json(raw: str) -> dict[str, Any]:
    log.debug("Parsing LLM JSON | raw_len=%s raw=%r", len(raw), raw[:800])
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


def _nonempty_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _merge_capture(prior: dict[str, Any], incoming: Any) -> dict[str, Any]:
    out = dict(prior or {})
    if isinstance(incoming, dict):
        for key, value in incoming.items():
            if value is not None:
                out[key] = value
    return out


def _snapshot(state: IntakeState) -> dict[str, Any]:
    return {
        "identity_verified": state.get("identity_verified"),
        "recording_notice_delivered": state.get("recording_notice_delivered"),
        "caller_name": state.get("caller_name"),
        "relationship_to_child": state.get("relationship_to_child"),
        "callback_number": state.get("callback_number"),
        "child_first_name": state.get("child_first_name"),
        "child_last_name": state.get("child_last_name"),
        "child_dob": state.get("child_dob"),
        "child_age_computed": state.get("child_age_computed"),
        "home_city": state.get("home_city"),
        "home_zip": state.get("home_zip"),
        "diagnosis_stated": state.get("diagnosis_stated"),
        "asd_diagnosis": state.get("asd_diagnosis"),
        "insurance_carrier": state.get("insurance_carrier"),
        "primary_disposition": state.get("primary_disposition"),
        "intake_complete": state.get("intake_complete"),
        "pending_field": state.get("pending_field"),
        "capture": state.get("capture") or {},
    }


def _build_handoff_summary(state: IntakeState, data: dict[str, Any]) -> str:
    provided = _nonempty_text(data.get("handoff_summary"))
    if provided:
        return provided
    parts: list[str] = []
    caller = state.get("caller_name") or data.get("caller_name")
    child = state.get("child_first_name") or data.get("child_first_name")
    dx = state.get("diagnosis_stated") or data.get("diagnosis_stated")
    carrier = state.get("insurance_carrier") or data.get("insurance_carrier")
    if caller:
        parts.append(f"Caller: {caller}")
    if child:
        parts.append(f"Child: {child}")
    if dx:
        parts.append(f"Dx stated: {dx}")
    if carrier:
        parts.append(f"Coverage: {carrier}")
    if not parts:
        return "Caller requested a human during Ava intake; little context collected yet."
    return "Ava intake summary — " + "; ".join(parts) + "."


def _field_is_collected(
    field: str,
    *,
    caller_name: str | None,
    relationship: str | None,
    callback: str | None,
    child_first: str | None,
    child_last: str | None,
    child_dob: str | None,
    home_city: str | None,
    home_zip: str | None,
    diagnosis: str | None,
    carrier: str | None,
    capture: dict[str, Any],
) -> bool:
    """Return whether a checklist field was answered or explicitly unavailable."""
    skipped = set(capture.get("skipped_fields") or [])
    if field in skipped:
        return True
    values = {
        "caller_name": bool(caller_name),
        "relationship": bool(relationship),
        "callback": bool(callback),
        "best_callback_time": bool(capture.get("best_callback_time")),
        "child_name": bool(child_first and child_last),
        "child_dob": bool(child_dob),
        "location": bool(home_city and home_zip),
        "language": bool(capture.get("preferred_language"))
        and "interpreter_needed" in capture,
        "diagnosis": bool(diagnosis),
        "diagnosing_provider": bool(
            capture.get("diagnosing_provider") and capture.get("diagnosis_date")
        ),
        "insurance": bool(carrier),
        "insurance_plan": bool(capture.get("insurance_plan")),
        "member_id": bool(
            capture.get("member_id") and capture.get("subscriber_name")
        ),
        "services": bool(capture.get("services_requested")),
        "availability": bool(
            capture.get("availability") and capture.get("care_setting")
        ),
        "consent": "contact_consent" in capture
        and (
            capture.get("contact_consent") is False
            or bool(capture.get("safe_contact_method"))
        ),
        "additional_notes": "additional_notes" in capture,
    }
    return values.get(field, False)


def _next_required_field(**details: Any) -> str:
    for field in _FIELD_SKIP_ORDER:
        if field == "close":
            return "close"
        if not _field_is_collected(field, **details):
            return field
    return "close"


def _graceful_close_reply(*, caller_name: str | None, child_first: str | None) -> str:
    who = child_first or "your child"
    thanks = f"Thanks{', ' + caller_name if caller_name else ''}."
    return (
        f"{thanks} Here's what happens next. Our benefits team reviews coverage "
        f"for {who}, and someone calls you back within {LIVE_FACTS['BENEFITS_SLA']}. "
        "I've saved the intake details for the care team. Thank you, and goodbye."
    )


def _is_repeat_ask(reply: str) -> bool:
    low = reply.lower()
    return any(
        p in low
        for p in (
            "didn't quite catch",
            "did not catch",
            "could you please repeat",
            "can you repeat",
            "say that again",
            "could you repeat",
            "i didn't catch",
            "i could not understand",
            "please repeat",
        )
    )


def _next_field_after(pending: str | None) -> str | None:
    if not pending:
        return "close"
    try:
        idx = _FIELD_SKIP_ORDER.index(pending)
    except ValueError:
        return "close"
    if idx + 1 >= len(_FIELD_SKIP_ORDER):
        return None
    return _FIELD_SKIP_ORDER[idx + 1]

def _apply_verification(state: IntakeState, data: dict[str, Any]) -> tuple[bool, str, str | None]:
    """Return (verified, continuity_block, possibly_updated_user_id)."""
    already = bool(state.get("identity_verified"))
    user_id = state.get("user_id")
    if already and user_id:
        return True, state.get("continuity_block") or "", user_id

    if not user_id:
        return False, state.get("continuity_block") or "", user_id

    caller_name = _nonempty_text(data.get("caller_name")) or state.get("caller_name")
    child_dob = _nonempty_text(data.get("child_dob")) or state.get("child_dob")
    verify_ready = bool(data.get("verify_ready")) or bool(caller_name and child_dob)

    db = SessionLocal()
    try:
        user = crepo.get_user(db, user_id)
        profile = crepo.get_intake_profile(db, user_id)
        if is_verified(user):
            block = build_continuity_block(db, user_id, verified=True)
            return True, block, user_id

        # Cross-channel: name+DOB may match a profile created via email
        matched = crepo.find_user_by_caller_and_child_dob(
            db, caller_name=caller_name, child_dob=child_dob
        )
        if matched and matched.id != user_id:
            session = repo.get_by_call_sid(db, state["call_sid"])
            phone = (session.caller_number if session else None) or state.get(
                "callback_number"
            )
            if phone:
                crepo.merge_channel_onto_user(
                    db,
                    from_user_id=user_id,
                    to_user_id=matched.id,
                    channel="phone",
                    value=phone,
                )
            user_id = matched.id
            repo.set_call_user_id(db, call_sid=state["call_sid"], user_id=user_id)
            profile = crepo.get_intake_profile(db, user_id)
            crepo.set_verification_status(
                db, user_id, status="verified", method="name_dob_cross_channel"
            )
            block = build_continuity_block(db, user_id, verified=True)
            log.info("Cross-channel identity merge on call | user_id=%s", user_id)
            return True, block, user_id

        if verify_ready and try_verify_identity(
            profile=profile,
            caller_name=caller_name,
            child_dob=child_dob,
            contact_match=True,
        ):
            crepo.set_verification_status(
                db, user_id, status="verified", method="name_dob_contact"
            )
            block = build_continuity_block(db, user_id, verified=True)
            log.info("Identity verified on call | user_id=%s", user_id)
            return True, block, user_id

        has_history = bool(crepo.get_recent_interactions(db, user_id, limit=1)) or (
            profile is not None
            and (profile.diagnosis_stated or profile.primary_disposition)
        )
        block = build_continuity_block(db, user_id, verified=False)
        if has_history:
            return False, block, user_id
        return False, block, user_id
    finally:
        db.close()


def extract_and_validate(state: IntakeState) -> IntakeState:
    log.debug(
        "NODE extract_and_validate | call_sid=%s user_id=%s user_text=%r pending=%r",
        state["call_sid"],
        state.get("user_id"),
        state["user_text"],
        state.get("pending_field"),
    )
    user_prompt = build_user_prompt(
        user_text=state["user_text"],
        state_snapshot=_snapshot(state),
        continuity_block=state.get("continuity_block") or "",
        history=state.get("messages") or [],
    )
    try:
        raw = chat_completion(
            [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ]
        )
        data = _parse_llm_json(raw)
    except Exception:
        # Never stall on LLM/STT garbage — skip current field and keep moving
        log.exception("LLM extract failed — advancing with safe defaults")
        next_field = _next_field_after(state.get("pending_field"))
        skipped_fields = list((state.get("capture") or {}).get("skipped_fields") or [])
        current_field = state.get("pending_field")
        if current_field and current_field not in {"recording_notice", "close"}:
            skipped_fields.append(current_field)
        if next_field in (None, "close"):
            data = {
                "reply": "",
                "pending_field": None,
                "capture": {"skipped_fields": skipped_fields},
                "primary_disposition": "Callback Queue",
                "is_complete": True,
                "should_end": True,
                "handoff_requested": False,
            }
        else:
            data = {
                "reply": "",
                "pending_field": next_field,
                "capture": {"skipped_fields": skipped_fields},
                "is_complete": False,
                "should_end": False,
                "handoff_requested": False,
            }

    verified, continuity_block, resolved_user_id = _apply_verification(state, data)

    def pick(field: str) -> Any:
        incoming = data.get(field)
        if incoming is None or incoming == "":
            return state.get(field)  # type: ignore[return-value]
        return incoming

    caller_name = _nonempty_text(pick("caller_name"))
    relationship = _nonempty_text(pick("relationship_to_child"))
    callback = _nonempty_text(pick("callback_number"))
    child_first = _nonempty_text(pick("child_first_name"))
    child_last = _nonempty_text(pick("child_last_name"))
    child_dob = _nonempty_text(pick("child_dob"))
    child_age = _nonempty_text(pick("child_age_computed"))
    home_city = _nonempty_text(pick("home_city"))
    home_zip = _nonempty_text(pick("home_zip"))
    diagnosis = _nonempty_text(pick("diagnosis_stated"))
    asd = data.get("asd_diagnosis")
    if asd is None:
        asd = state.get("asd_diagnosis")
    carrier = _nonempty_text(pick("insurance_carrier"))

    # If model blanked insurance but caller just answered an insurance ask, accept utterance.
    # Prefer explicit pending_field from the model (including null = done) over prior state.
    if "pending_field" in data:
        pending_in = data.get("pending_field")
    else:
        pending_in = state.get("pending_field")
    user_utt = (state.get("user_text") or "").strip()
    if not carrier and pending_in == "insurance" and user_utt:
        # Avoid accepting meta-answers like "what?" / pure fillers
        low = user_utt.lower()
        if low not in {"what", "huh", "sorry", "repeat", "pardon"} and len(user_utt) >= 2:
            carrier = user_utt.rstrip(".")

    # Guard: do not overwrite adult caller_name with the child's name.
    prior_caller = state.get("caller_name")
    if (
        caller_name
        and child_first
        and caller_name.strip().lower() == child_first.strip().lower()
        and prior_caller
        and prior_caller.strip().lower() != child_first.strip().lower()
    ):
        log.warning(
            "Rejected caller_name==child_first_name mixup | kept caller=%r child=%r",
            prior_caller,
            child_first,
        )
        caller_name = prior_caller
    if (
        caller_name
        and child_first
        and caller_name.strip().lower() == child_first.strip().lower()
        and relationship
        and relationship.strip().lower()
        not in {"self", "myself", "patient", "me"}
    ):
        # Relative calling: prefer keeping prior caller if any, else clear mistaken overwrite
        if prior_caller and prior_caller.strip().lower() != child_first.strip().lower():
            caller_name = prior_caller
        elif state.get("caller_name"):
            caller_name = state.get("caller_name")

    disposition = _nonempty_text(data.get("primary_disposition")) or state.get(
        "primary_disposition"
    )
    if disposition and disposition not in _DISPOSITIONS:
        disposition = state.get("primary_disposition")
    intake_complete = data.get("intake_complete")
    if intake_complete is None:
        intake_complete = state.get("intake_complete")

    capture = _merge_capture(state.get("capture") or {}, data.get("capture"))
    checklist_details = {
        "caller_name": caller_name,
        "relationship": relationship,
        "callback": callback,
        "child_first": child_first,
        "child_last": child_last,
        "child_dob": child_dob,
        "home_city": home_city,
        "home_zip": home_zip,
        "diagnosis": diagnosis,
        "carrier": carrier,
        "capture": capture,
    }
    recording_notice = bool(
        data.get("recording_notice_delivered")
        or state.get("recording_notice_delivered")
    )

    handoff_requested = bool(data.get("handoff_requested"))
    handoff_reason = _nonempty_text(data.get("handoff_reason")) or ""
    handoff_summary = ""
    should_end = bool(data.get("should_end"))
    is_complete = False
    pending = data.get("pending_field")
    turn_count = int(state.get("turn_count") or 0) + 1
    unclear_streak = int(state.get("unclear_streak") or 0)
    prior_pending = state.get("pending_field")

    if handoff_requested:
        should_end = True
        is_complete = False
        pending = None
        unclear_streak = 0
        if not handoff_reason:
            handoff_reason = "caller requested human agent"
        if not disposition:
            disposition = "Warm Transfer"
        handoff_summary = _build_handoff_summary(state, data)
        reply = _nonempty_text(data.get("reply")) or _DEFAULT_HANDOFF_REPLY
    else:
        reply = str(data.get("reply") or "")
        is_repeat_ask = _is_repeat_ask(reply) if reply else False

        # Track unclear retries on the same pending field
        if is_repeat_ask and pending_in and pending_in == prior_pending:
            unclear_streak += 1
        elif pending_in != prior_pending:
            unclear_streak = 0

        # After one unclear retry, skip the field and move on
        if unclear_streak > _MAX_UNCLEAR and pending_in not in {None, "close"}:
            skipped = pending_in
            skipped_fields = list(capture.get("skipped_fields") or [])
            if skipped not in skipped_fields:
                skipped_fields.append(skipped)
            capture["skipped_fields"] = skipped_fields
            pending = _next_required_field(**checklist_details)
            unclear_streak = 0
            log.info("Skipping unclear field | skipped=%s next=%s", skipped, pending)
            if pending == "close" or pending is None:
                disposition = disposition or "Callback Queue"
                is_complete = True
                should_end = True
                reply = _graceful_close_reply(
                    caller_name=caller_name, child_first=child_first
                )
            else:
                reply = (
                    "No problem, we'll leave that for the care team to confirm. "
                    + _FIELD_PROMPTS.get(
                        str(pending),
                        "Let's keep going. What else should I capture for intake?",
                    )
                )

        # Accept insurance answers without re-asking, then continue the checklist.
        elif carrier and (
            pending_in == "insurance"
            or (is_repeat_ask and prior_pending == "insurance")
        ):
            unclear_streak = 0
            pending = _next_required_field(**checklist_details)
            if pending == "close":
                disposition = disposition or "Callback Queue"
                is_complete = True
                should_end = True
                pending = None
                reply = _graceful_close_reply(
                    caller_name=caller_name, child_first=child_first
                )
            else:
                reply = f"Got it — {carrier}. " + _FIELD_PROMPTS[pending]

        else:
            # A normal close is allowed only after every checklist field was handled.
            model_complete = bool(data.get("is_complete") and disposition)
            next_required = _next_required_field(**checklist_details)
            checklist_complete = next_required == "close"
            if checklist_complete:
                disposition = disposition or "Callback Queue"
                is_complete = True
                should_end = True
                pending = None
                unclear_streak = 0
                if intake_complete is None:
                    intake_complete = True
                reply = _graceful_close_reply(
                    caller_name=caller_name, child_first=child_first
                )
            else:
                # Override premature model closure and keep the interview flowing.
                should_end = False
                is_complete = False
                disposition = None
                pending = next_required
                if (
                    not reply
                    or data.get("should_end")
                    or data.get("is_complete")
                    or pending_in != next_required
                ):
                    reply = _FIELD_PROMPTS[next_required]

        # Hard turn cap — never leave a call hanging
        if not should_end and turn_count >= _MAX_TURNS:
            disposition = disposition or "Callback Queue"
            is_complete = True
            should_end = True
            pending = None
            reply = _graceful_close_reply(
                caller_name=caller_name, child_first=child_first
            )
            log.info("Force-close on turn cap | turn_count=%s", turn_count)

    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": state["user_text"]})
    messages.append({"role": "assistant", "content": reply})

    # Legacy mirrors for CallSession columns / debug API
    patient_name = caller_name
    patient_age = None
    diseases = diagnosis
    medications = carrier

    updated: IntakeState = {
        "call_sid": state["call_sid"],
        "user_id": resolved_user_id or state.get("user_id"),
        "user_text": state["user_text"],
        "messages": messages,
        "identity_verified": verified,
        "continuity_block": continuity_block,
        "recording_notice_delivered": recording_notice,
        "caller_name": caller_name,
        "relationship_to_child": relationship,
        "callback_number": callback,
        "child_first_name": child_first,
        "child_last_name": child_last,
        "child_dob": child_dob,
        "child_age_computed": child_age,
        "home_city": home_city,
        "home_zip": home_zip,
        "diagnosis_stated": diagnosis,
        "asd_diagnosis": asd if isinstance(asd, bool) else None,
        "insurance_carrier": carrier,
        "primary_disposition": disposition,
        "intake_complete": intake_complete if isinstance(intake_complete, bool) else None,
        "capture": capture,
        "pending_field": pending,  # type: ignore[typeddict-item]
        "unclear_streak": unclear_streak,
        "turn_count": turn_count,
        "reply": reply,
        "is_complete": is_complete,
        "should_end": should_end,
        "handoff_requested": handoff_requested,
        "handoff_reason": handoff_reason,
        "handoff_summary": handoff_summary,
        "validation_notes": str(data.get("validation_notes") or ""),
        "patient_name": patient_name,
        "patient_age": patient_age,
        "ready_to_proceed": True if is_complete else state.get("ready_to_proceed"),
        "diseases": diseases,
        "medications": medications,
    }
    log.info(
        "extract_and_validate done | complete=%s should_end=%s handoff=%s disposition=%s "
        "verified=%s child=%r reply=%r",
        updated["is_complete"],
        updated["should_end"],
        updated["handoff_requested"],
        updated["primary_disposition"],
        updated["identity_verified"],
        updated["child_first_name"],
        updated["reply"],
    )
    return updated


def save_to_db(state: IntakeState) -> IntakeState:
    log.debug("NODE save_to_db | call_sid=%s user_id=%s", state["call_sid"], state.get("user_id"))
    if state.get("handoff_requested"):
        status = "handoff_pending"
    elif state.get("is_complete"):
        status = "complete"
    else:
        status = "in_progress"

    db = SessionLocal()
    try:
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in (state.get("messages") or [])
        )
        repo.update_intake(
            db,
            call_sid=state["call_sid"],
            patient_name=state.get("patient_name") or state.get("caller_name"),
            patient_age=state.get("patient_age"),
            ready_to_proceed=state.get("ready_to_proceed"),
            diseases=state.get("diseases") or state.get("diagnosis_stated"),
            medications=state.get("medications") or state.get("insurance_carrier"),
            transcript=transcript,
            status=status,
            handoff_reason=state.get("handoff_reason") or None,
            handoff_summary=state.get("handoff_summary") or None,
        )

        user_id = state.get("user_id")
        if user_id:
            capture = dict(state.get("capture") or {})
            profile_fields = {
                "caller_name": state.get("caller_name"),
                "relationship_to_child": state.get("relationship_to_child"),
                "callback_number": state.get("callback_number"),
                "child_first_name": state.get("child_first_name"),
                "child_last_name": state.get("child_last_name"),
                "child_dob": state.get("child_dob"),
                "child_age_computed": state.get("child_age_computed"),
                "home_city": state.get("home_city"),
                "home_zip": state.get("home_zip"),
                "diagnosis_stated": state.get("diagnosis_stated"),
                "asd_diagnosis": state.get("asd_diagnosis"),
                "insurance_carrier": state.get("insurance_carrier"),
                "primary_disposition": state.get("primary_disposition"),
                "intake_complete": state.get("intake_complete"),
            }
            # Always upsert profile mid-call; append interaction only on close-ish events
            crepo.upsert_intake_profile(
                db,
                user_id,
                capture_delta=capture or None,
                **{k: v for k, v in profile_fields.items() if v is not None},
            )
            if (
                state.get("is_complete")
                or state.get("should_end")
                or state.get("handoff_requested")
                or state.get("primary_disposition")
            ):
                summary = summarize_interaction_from_capture(
                    channel="phone",
                    disposition=state.get("primary_disposition"),
                    capture={
                        **capture,
                        **{k: v for k, v in profile_fields.items() if v is not None},
                    },
                    reply_excerpt=state.get("reply"),
                )
                persist_channel_turn(
                    db,
                    user_id=user_id,
                    channel="phone",
                    external_id=state["call_sid"],
                    summary=summary,
                    outcome=state.get("primary_disposition") or status,
                    verification_status=(
                        "verified" if state.get("identity_verified") else "unverified"
                    ),
                    profile_fields=profile_fields,
                    capture_delta=capture,
                )
            if state.get("callback_number"):
                crepo.link_channel_identity(
                    db,
                    user_id=user_id,
                    channel="phone",
                    value=state["callback_number"],
                )
    finally:
        db.close()
    log.info("save_to_db committed | call_sid=%s status=%s", state["call_sid"], status)
    return state


def route_after_extract(state: IntakeState) -> Literal["save_to_db", "__end__"]:
    if (
        state.get("is_complete")
        or state.get("should_end")
        or state.get("handoff_requested")
        or state.get("caller_name")
        or state.get("child_first_name")
    ):
        return "save_to_db"
    return "__end__"


def build_intake_graph():
    log.debug("Compiling LangGraph Ava intake graph")
    graph = StateGraph(IntakeState)
    graph.add_node("extract_and_validate", extract_and_validate)
    graph.add_node("save_to_db", save_to_db)
    graph.set_entry_point("extract_and_validate")
    graph.add_conditional_edges(
        "extract_and_validate",
        route_after_extract,
        {
            "save_to_db": "save_to_db",
            "__end__": END,
        },
    )
    graph.add_edge("save_to_db", END)
    compiled = graph.compile()
    log.info("LangGraph Ava intake graph compiled")
    return compiled


_GRAPH = None


def get_intake_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_intake_graph()
    return _GRAPH


def _initial_state(call_sid: str, user_text: str) -> IntakeState:
    user_id: str | None = None
    continuity_block = "CONTINUITY: no linked user yet."
    identity_verified = False
    db = SessionLocal()
    try:
        session = repo.get_by_call_sid(db, call_sid)
        if session and session.user_id:
            user_id = session.user_id
        elif session and session.caller_number:
            user = crepo.find_or_create_user_by_phone(db, session.caller_number)
            if user:
                user_id = user.id
                repo.set_call_user_id(db, call_sid=call_sid, user_id=user_id)
        if user_id:
            user = crepo.get_user(db, user_id)
            identity_verified = is_verified(user)
            continuity_block = build_continuity_block(
                db, user_id, verified=identity_verified
            )
    finally:
        db.close()

    return {
        "call_sid": call_sid,
        "user_id": user_id,
        "user_text": user_text,
        "messages": [],
        "identity_verified": identity_verified,
        "continuity_block": continuity_block,
        "recording_notice_delivered": False,
        "caller_name": None,
        "relationship_to_child": None,
        "callback_number": None,
        "child_first_name": None,
        "child_last_name": None,
        "child_dob": None,
        "child_age_computed": None,
        "home_city": None,
        "home_zip": None,
        "diagnosis_stated": None,
        "asd_diagnosis": None,
        "insurance_carrier": None,
        "primary_disposition": None,
        "intake_complete": None,
        "capture": {},
        "pending_field": "recording_notice",
        "unclear_streak": 0,
        "turn_count": 0,
        "reply": "",
        "is_complete": False,
        "should_end": False,
        "handoff_requested": False,
        "handoff_reason": "",
        "handoff_summary": "",
        "validation_notes": "",
        "patient_name": None,
        "patient_age": None,
        "ready_to_proceed": None,
        "diseases": None,
        "medications": None,
    }


def seed_prior_after_greeting(
    call_sid: str, greeting: str | None = None
) -> IntakeState:
    """Seed state after the Ava opening greeting (recording notice already spoken)."""
    state = _initial_state(call_sid, user_text="")
    state["recording_notice_delivered"] = True
    state["pending_field"] = "caller_name"
    state["messages"] = [
        {"role": "assistant", "content": greeting or "How may I help with intake today?"}
    ]
    return state


def run_intake_turn(
    *,
    call_sid: str,
    user_text: str,
    prior: IntakeState | None = None,
) -> IntakeState:
    log.info("run_intake_turn | call_sid=%s user_text=%r", call_sid, user_text)
    graph = get_intake_graph()
    if prior is None:
        state = _initial_state(call_sid, user_text)
    else:
        state = {
            **prior,
            "call_sid": call_sid,
            "user_text": user_text,
            "handoff_requested": False,
            "handoff_reason": "",
            "handoff_summary": "",
        }
    result = graph.invoke(state)
    log.debug("run_intake_turn result | %s", result)
    return result  # type: ignore[return-value]
