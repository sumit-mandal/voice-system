""" State for email lanngraph agent""" 

from __future__ import annotations 
from typing import Any, Literal, TypedDict 

EmailIntent = Literal[
    "history_records",
    "appointment",
    "billing",
    "general_question",
    "unclear",
    "escalate",
]

class EmailState(TypedDict):
    message_id:str 
    from_address:str
    subject:str 
    body_text:str  

    intent: EmailIntent | None 
    intent_confidence: float
    patient_id : str | None 

    tool_name: str | None 
    tool_result: dict[str, Any] | None  

    reply_subject: str
    reply_body : str 
    needs_clarification: bool
    should_escalate: bool
    error: str