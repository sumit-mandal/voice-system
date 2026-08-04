"""Prompts for Ava phone intake LangGraph agent."""

from __future__ import annotations

import json
from typing import Any

from app.continuity.ava.policy import (
    LIVE_FACTS,
    RECORDING_NOTICE,
    build_ava_phone_addendum,
    build_ava_shared_policy,
)
from app.db.clinic_repo import get_clinic_name


def build_system_prompt(clinic_name: str | None = None) -> str:
    name = clinic_name or get_clinic_name()
    return f"""{build_ava_shared_policy(name)}

{build_ava_phone_addendum(name)}

Phone turn protocol:
1) If recording_notice_delivered is false, your FIRST spoken content must include the
   exact recording notice once, then the natural Ava open. Set recording_notice_delivered
   true in JSON after delivering it.
2) Collect fields one at a time per capture order. Prefer pending_field guidance.
3) If CONTINUITY says prior contact may exist and identity_verified is false, verify
   before disclosing prior clinical/schedule/coverage details.
4) When enough R fields are collected for the scenario, set primary_disposition and
   close with ACTION + OWNER + response window (use live facts).
5) Human handoff: if caller asks for a human, or verification fails / authority contested,
   set handoff_requested=true (Warm Transfer / Verification Failed as appropriate).
6) NAME HANDLING (strict): Never ask the caller to spell their name or the child's
   name letter by letter. Accept a normal spoken name. Confirm with a short read-back
   once, then advance. If STT looks garbled, ask them to repeat the name once — not spell it.

Return ONLY valid JSON:
{{
  "caller_name": string | null,
  "relationship_to_child": string | null,
  "callback_number": string | null,
  "child_first_name": string | null,
  "child_last_name": string | null,
  "child_dob": string | null,
  "child_age_computed": string | null,
  "home_city": string | null,
  "home_zip": string | null,
  "diagnosis_stated": string | null,
  "asd_diagnosis": boolean | null,
  "insurance_carrier": string | null,
  "primary_disposition": "Scheduled" | "Warm Transfer" | "Callback Queue" | "Records Needed" | "Referred Out" | "Waitlist" | "Safety Stop" | "Verification Failed" | null,
  "intake_complete": boolean | null,
  "capture": object,
  "recording_notice_delivered": boolean,
  "identity_verified": boolean,
  "verify_ready": boolean,
  "pending_field": string | null,
  "validation_notes": string,
  "reply": string,
  "is_complete": boolean,
  "should_end": boolean,
  "handoff_requested": boolean,
  "handoff_reason": string,
  "handoff_summary": string
}}

capture may include: member_id, services_requested, speech_requested, medicaid_pihp_status,
eval_referral_sent, records_pending, employer_size_flag, secondary_tasks, scenario_id,
diagnosing_provider, etc.

Live facts: BENEFITS_SLA={LIVE_FACTS['BENEFITS_SLA']}; ABA_ASSESSMENT_WAIT={LIVE_FACTS['ABA_ASSESSMENT_WAIT']}; SPEECH_OT_WAIT={LIVE_FACTS['SPEECH_OT_WAIT']}.
Recording notice exact text: "{RECORDING_NOTICE}"
Clinic name (use exactly): "{name}"
"""


def __getattr__(name: str) -> str:
    if name == "SYSTEM_PROMPT":
        return build_system_prompt()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_user_prompt(
    *,
    user_text: str,
    state_snapshot: dict[str, Any],
    continuity_block: str,
    history: list[dict[str, str]],
) -> str:
    history_lines = []
    for turn in history[-10:]:
        history_lines.append(f"{turn['role'].upper()}: {turn['content']}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    return f"""{continuity_block}

Known Ava slots so far:
{json.dumps(state_snapshot, indent=2, default=str)}

Recent conversation:
{history_block}

Latest caller utterance:
{user_text!r}

Respond with JSON only.
"""
