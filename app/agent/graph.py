"""LangGraph intake graph: extract/validate → save when complete or not-ready."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.agent.llm import chat_completion
from app.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from app.agent.state import IntakeState
from app.db import repository as repo
from app.db.session import SessionLocal
from app.logging_setup import get_logger

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_PENDING = {"name", "age", "ready", "diseases", "medications", None}


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


def extract_and_validate(state: IntakeState) -> IntakeState:
    log.debug(
        "NODE extract_and_validate | call_sid=%s user_text=%r pending=%r",
        state["call_sid"],
        state["user_text"],
        state.get("pending_field"),
    )
    user_prompt = build_user_prompt(
        user_text=state["user_text"],
        patient_name=state.get("patient_name"),
        patient_age=state.get("patient_age"),
        ready_to_proceed=state.get("ready_to_proceed"),
        diseases=state.get("diseases"),
        medications=state.get("medications"),
        pending_field=state.get("pending_field"),
        history=state.get("messages") or [],
    )
    log.debug("Invoking Gemini chat_completion | prompt_chars=%s", len(user_prompt))
    raw = chat_completion(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    log.debug("LLM raw response | %r", raw[:1200])
    data = _parse_llm_json(raw)

    name = data.get("patient_name") or state.get("patient_name")
    age = data.get("patient_age")
    if age is None:
        age = state.get("patient_age")
    elif isinstance(age, str) and age.isdigit():
        age = int(age)

    name_valid = bool(data.get("name_valid"))
    age_valid = bool(data.get("age_valid"))
    if isinstance(age, int) and not (0 <= age <= 120):
        log.warning("Age out of range from LLM | age=%s — forcing invalid", age)
        age_valid = False
        age = state.get("patient_age")

    ready = data.get("ready_to_proceed")
    if ready is None:
        ready = state.get("ready_to_proceed")
    ready_valid = bool(data.get("ready_valid")) and ready is not None

    diseases = _nonempty_text(data.get("diseases")) or state.get("diseases")
    medications = _nonempty_text(data.get("medications")) or state.get("medications")
    diseases_valid = bool(data.get("diseases_valid")) and bool(diseases)
    medications_valid = bool(data.get("medications_valid")) and bool(medications)

    should_end = bool(data.get("should_end"))
    # Caller declined to proceed — end after goodbye.
    if ready_valid and ready is False:
        should_end = True
        is_complete = False
        pending = None
        log.info("Caller not ready to proceed — will end call")
    else:
        is_complete = bool(
            data.get("is_complete")
            and name_valid
            and age_valid
            and ready_valid
            and ready is True
            and diseases_valid
            and medications_valid
            and name
            and age is not None
            and diseases
            and medications
        )
        pending = data.get("pending_field")
        if is_complete:
            pending = None
            should_end = True
        elif pending not in _PENDING:
            if not name_valid or not name:
                pending = "name"
            elif not age_valid or age is None:
                pending = "age"
            elif not ready_valid or ready is None:
                pending = "ready"
            elif not diseases_valid:
                pending = "diseases"
            elif not medications_valid:
                pending = "medications"
            else:
                pending = None

    messages = list(state.get("messages") or [])
    messages.append({"role": "user", "content": state["user_text"]})
    reply = str(data.get("reply") or "Could you please repeat that?")
    messages.append({"role": "assistant", "content": reply})

    updated: IntakeState = {
        "call_sid": state["call_sid"],
        "user_text": state["user_text"],
        "messages": messages,
        "patient_name": name if name_valid else state.get("patient_name"),
        "patient_age": age if age_valid else state.get("patient_age"),
        "ready_to_proceed": ready if ready_valid else state.get("ready_to_proceed"),
        "diseases": diseases if diseases_valid else state.get("diseases"),
        "medications": medications if medications_valid else state.get("medications"),
        "pending_field": pending,  # type: ignore[typeddict-item]
        "reply": reply,
        "is_complete": is_complete,
        "should_end": should_end,
        "validation_notes": str(data.get("validation_notes") or ""),
    }
    log.info(
        "extract_and_validate done | complete=%s should_end=%s pending=%s "
        "name=%r age=%r ready=%r diseases=%r meds=%r reply=%r",
        updated["is_complete"],
        updated["should_end"],
        updated["pending_field"],
        updated["patient_name"],
        updated["patient_age"],
        updated["ready_to_proceed"],
        updated["diseases"],
        updated["medications"],
        updated["reply"],
    )
    return updated


def save_to_db(state: IntakeState) -> IntakeState:
    log.debug("NODE save_to_db | call_sid=%s", state["call_sid"])
    if state.get("is_complete"):
        status = "complete"
    elif state.get("ready_to_proceed") is False:
        status = "not_ready"
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
            patient_name=state.get("patient_name"),
            patient_age=state.get("patient_age"),
            ready_to_proceed=state.get("ready_to_proceed"),
            diseases=state.get("diseases"),
            medications=state.get("medications"),
            transcript=transcript,
            status=status,
        )
    finally:
        db.close()
    log.info("save_to_db committed | call_sid=%s status=%s", state["call_sid"], status)
    return state


def route_after_extract(state: IntakeState) -> Literal["save_to_db", "__end__"]:
    # Persist whenever we end OR whenever we have any new slots mid-call.
    if state.get("is_complete") or state.get("should_end") or state.get("patient_name"):
        log.debug(
            "Routing → save_to_db | complete=%s should_end=%s",
            state.get("is_complete"),
            state.get("should_end"),
        )
        return "save_to_db"
    log.debug("Routing → END")
    return "__end__"


def build_intake_graph():
    log.debug("Compiling LangGraph intake graph")
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
    log.info("LangGraph intake graph compiled")
    return compiled


_GRAPH = None


def get_intake_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_intake_graph()
    return _GRAPH


def run_intake_turn(
    *,
    call_sid: str,
    user_text: str,
    prior: IntakeState | None = None,
) -> IntakeState:
    log.info("run_intake_turn | call_sid=%s user_text=%r", call_sid, user_text)
    graph = get_intake_graph()
    if prior is None:
        state: IntakeState = {
            "call_sid": call_sid,
            "user_text": user_text,
            "messages": [],
            "patient_name": None,
            "patient_age": None,
            "ready_to_proceed": None,
            "diseases": None,
            "medications": None,
            "pending_field": "name",
            "reply": "",
            "is_complete": False,
            "should_end": False,
            "validation_notes": "",
        }
    else:
        state = {
            **prior,
            "call_sid": call_sid,
            "user_text": user_text,
        }
    result = graph.invoke(state)
    log.debug("run_intake_turn result | %s", result)
    return result  # type: ignore[return-value]
