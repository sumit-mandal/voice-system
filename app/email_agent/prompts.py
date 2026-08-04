"""Prompts for Ava email intent + reply drafting."""

from __future__ import annotations

import json

from app.continuity.ava.policy import (
    EMAIL_PRIVACY_FOOTER,
    LIVE_FACTS,
    build_ava_email_addendum,
    build_ava_shared_policy,
)
from app.db.clinic_repo import get_clinic_name


def build_classify_system(clinic_name: str | None = None) -> str:
    name = clinic_name or get_clinic_name()
    return f"""{build_ava_shared_policy(name)}

{build_ava_email_addendum()}

Classify inbound {name} family emails for Ava.
Return ONLY JSON:
{{
  "intent": "new_intake" | "continue_intake" | "records_request" | "benefits_status" | "human_handoff" | "unclear" | "escalate",
  "intent_confidence": number,
  "needs_clarification": boolean,
  "should_escalate": boolean,
  "caller_name": string | null,
  "child_first_name": string | null,
  "child_dob": string | null,
  "diagnosis_stated": string | null,
  "insurance_carrier": string | null,
  "notes": string
}}

Rules:
- new_intake: first contact / starting services for a child.
- continue_intake: follow-up on an existing intake thread or channel switch.
- records_request: evaluation reports / forms (route to secure workflow language).
- benefits_status: coverage / cost / Medicaid / PIHP questions.
- human_handoff / escalate: angry, legal, emergency, urgent safety, or explicit human request.
- unclear: cannot tell what they need.
- Never put clinical detail in notes that would later become a subject line.
- Use clinic name "{name}" exactly when naming the organization.
"""


def build_draft_system(clinic_name: str | None = None) -> str:
    name = clinic_name or get_clinic_name()
    return f"""{build_ava_shared_policy(name)}

{build_ava_email_addendum()}

Draft ONE Ava email reply for the channel.
- Specific subject WITHOUT diagnosis, member ID, or clinical detail.
- Short paragraphs + scannable action list.
- One primary purpose.
- Name action, owner, and response window (use live facts).
- Use clinic name "{name}" exactly when naming the organization.
- Append this privacy footer on its own final paragraph:
  {EMAIL_PRIVACY_FOOTER}
- Do not invent clinical data. Use continuity/tool facts only when identity_verified.
- If not verified and prior history may exist, ask for verification (caller name + child DOB)
  without confirming the child is a patient.

Return ONLY JSON:
{{
  "reply_subject": string,
  "reply_body": string,
  "primary_disposition": "Scheduled" | "Warm Transfer" | "Callback Queue" | "Records Needed" | "Referred Out" | "Waitlist" | "Safety Stop" | "Verification Failed" | null,
  "capture": object,
  "interaction_summary": string,
  "identity_verified": boolean
}}

Live facts: BENEFITS_SLA={LIVE_FACTS['BENEFITS_SLA']}; ABA_ASSESSMENT_WAIT={LIVE_FACTS['ABA_ASSESSMENT_WAIT']}; SPEECH_OT_WAIT={LIVE_FACTS['SPEECH_OT_WAIT']}.
"""


def __getattr__(name: str) -> str:
    if name == "CLASSIFY_SYSTEM":
        return build_classify_system()
    if name == "DRAFT_SYSTEM":
        return build_draft_system()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_classify_prompt(
    *,
    subject: str,
    body_text: str,
    from_address: str,
    continuity_block: str = "",
) -> str:
    return f"""{continuity_block}

FROM: {from_address}
SUBJECT: {subject}
BODY: {body_text}

Classify intent. JSON only.
"""


def build_draft_prompt(
    *,
    subject: str,
    body_text: str,
    intent: str,
    tool_result: dict | None,
    continuity_block: str,
    identity_verified: bool,
) -> str:
    return f"""{continuity_block}

identity_verified: {identity_verified}
Original subject: {subject}
Original body:
{body_text}
Intent: {intent}
Tool / continuity facts: {json.dumps(tool_result, default=str) if tool_result else None}

Draft the reply. JSON only.
"""
