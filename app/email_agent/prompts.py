"""Prompts for email intent + reply drafting."""

CLASSIFY_SYSTEM = """You classify inbound healthcare emails for an automation agent.
Return ONLY JSON:
{
  "intent": "history_records" | "appointment" | "billing" | "general_question" | "unclear" | "escalate",
  "intent_confidence": number,
  "patient_hint": string | null,
  "needs_clarification": boolean,
  "should_escalate": boolean,
  "notes": string
}  

Rules:
- history_records: ask for medical/visit/history records, past visits, charts.
- escalate: angry, legal, emergency, or explicitly wants a human.
- unclear: missing who they are or what they want.
"""

DRAFT_SYSTEM = """You draft a short, professional email reply.
Do not invent clinical data. Only use facts from tool_result / context.
Return ONLY JSON:
{
  "reply_subject": string,
  "reply_body": string
}
"""

def build_classify_prompt(*, subject:str,body_text:str, from_address:str) -> str:

    return f"""FROM: {from_address}
SUBJECT: {subject}
BODY: {body_text}

Classify intent. JSON only.
"""

def build_draft_prompt(
    *,
    subject: str,
    body_text: str,
    intent: str,
    tool_result: dict | None,) -> str:
    
    return f"""Original subject: {subject}
    Original body:
    {body_text}
    Intent: {intent}
    Tool result (facts only): {tool_result!r}
    Draft the reply. JSON only.
    """