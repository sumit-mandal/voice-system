"""Twilio voice webhooks — inbound, outbound dial, status → LiveKit SIP bridge."""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, Form, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from twilio.rest import Client
from twilio.twiml.voice_response import Dial, VoiceResponse

from app.config import get_settings
from app.db import repository as repo
from app.db.session import SessionLocal
from app.livekit_app.rooms import create_room, dispatch_agent, make_room_name
from app.logging_setup import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/twilio", tags=["twilio"])

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")
# Short-lived whisper text for human agents, keyed by original caller CallSid.
_HANDOFF_WHISPER: dict[str, str] = {}


class OutboundCallRequest(BaseModel):
    to: str = Field(..., description="Your phone in E.164, e.g. +9198XXXXXXXX")


class OutboundCallResponse(BaseModel):
    call_sid: str
    to: str
    from_number: str
    status: str
    answer_url: str


@router.post("/voice/inbound")
async def inbound_call(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(""),
    To: str = Form(""),
    Direction: str = Form(""),
) -> Response:
    """Answer URL for both inbound PSTN and outbound-api when the callee picks up."""
    settings = get_settings()
    log.info(
        "Twilio voice bridge | CallSid=%s From=%s To=%s Direction=%s",
        CallSid,
        From,
        To,
        Direction,
    )
    log.debug("Twilio headers | %s", dict(request.headers))

    # For outbound-api, "To" is the person we dialed; for inbound, "From" is the caller.
    party = To if Direction.startswith("outbound") else From
    return await _bridge_to_livekit(
        call_sid=CallSid,
        party_number=party or From or To,
        from_number=From,
        to_number=To,
        direction=Direction or "inbound",
    )


@router.post("/voice/outbound", response_model=OutboundCallResponse)
async def start_outbound_call(body: OutboundCallRequest) -> OutboundCallResponse:
    """Have Twilio dial `to`; when answered, bridge into LiveKit + agent."""
    settings = get_settings()
    to = body.to.strip().replace(" ", "")
    if not _E164.match(to):
        raise HTTPException(
            status_code=400,
            detail="to must be E.164, e.g. +9198XXXXXXXX",
        )
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        raise HTTPException(status_code=500, detail="Twilio credentials not configured")
    if not settings.twilio_phone_number:
        raise HTTPException(status_code=500, detail="TWILIO_PHONE_NUMBER not set")
    if not settings.livekit_sip_trunk_id:
        raise HTTPException(
            status_code=500,
            detail="LIVEKIT_SIP_TRUNK_ID must be set before placing calls",
        )

    base = settings.public_base_url.rstrip("/")
    answer_url = f"{base}/twilio/voice/inbound"
    status_url = f"{base}/twilio/voice/status"

    log.info(
        "Placing outbound call | to=%s from=%s answer_url=%s",
        to,
        settings.twilio_phone_number,
        answer_url,
    )

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    try:
        call = client.calls.create(
            to=to,
            from_=settings.twilio_phone_number,
            url=answer_url,
            method="POST",
            status_callback=status_url,
            status_callback_method="POST",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )
    except Exception as exc:
        log.exception("Twilio calls.create failed | to=%s", to)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    log.info("Outbound queued | CallSid=%s status=%s", call.sid, call.status)
    return OutboundCallResponse(
        call_sid=call.sid,
        to=to,
        from_number=settings.twilio_phone_number,
        status=call.status or "queued",
        answer_url=answer_url,
    )


async def _bridge_to_livekit(
    *,
    call_sid: str,
    party_number: str,
    from_number: str,
    to_number: str,
    direction: str,
) -> Response:
    settings = get_settings()
    if not settings.livekit_sip_trunk_id:
        log.error("LIVEKIT_SIP_TRUNK_ID is empty — cannot bridge Twilio → LiveKit SIP")
        raise RuntimeError("LIVEKIT_SIP_TRUNK_ID must be set for phone calling")

    room_name = make_room_name(call_sid)
    await create_room(room_name)

    db = SessionLocal()
    try:
        repo.create_call_session(
            db,
            call_sid=call_sid,
            room_name=room_name,
            caller_number=party_number or None,
        )
    finally:
        db.close()

    metadata = json.dumps(
        {
            "call_sid": call_sid,
            "from": from_number,
            "to": to_number,
            "direction": direction,
        }
    )
    await dispatch_agent(room_name, metadata)

    sip_host = settings.livekit_sip_host or _sip_host(settings.livekit_url)
    sip_uri = f"sip:{room_name}@{sip_host}"
    log.info(
        "Returning TwiML Dial SIP | uri=%s trunk=%s sip_host=%s",
        sip_uri,
        settings.livekit_sip_trunk_id,
        sip_host,
    )

    response = VoiceResponse()
    response.say("Connecting you to the intake assistant. One moment.")
    dial: Dial = Dial(answer_on_bridge=True)
    dial.sip(sip_uri)
    response.append(dial)
    twiml = str(response)
    log.debug("TwiML response | %s", twiml)
    return Response(content=twiml, media_type="application/xml")


def _sip_host(livekit_url: str) -> str:
    # wss://myproj.livekit.cloud → myproj.sip.livekit.cloud
    host = livekit_url.removeprefix("wss://").removeprefix("ws://").split("/")[0]
    if host.endswith(".livekit.cloud"):
        project = host.removesuffix(".livekit.cloud")
        sip_host = f"{project}.sip.livekit.cloud"
        log.debug("Derived SIP host=%s from LIVEKIT_URL=%s", sip_host, livekit_url)
        return sip_host
    log.debug("Using raw host for SIP | host=%s", host)
    return host


@router.post("/voice/status")
async def call_status(
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
) -> dict[str, str]:
    log.info("Twilio status callback | CallSid=%s CallStatus=%s", CallSid, CallStatus)
    if CallStatus in {"completed", "busy", "failed", "no-answer", "canceled"}:
        db = SessionLocal()
        try:
            row = repo.get_by_call_sid(db, CallSid)
            if row and row.status in {"in_progress", "handoff_pending"}:
                repo.update_intake(
                    db,
                    call_sid=CallSid,
                    patient_name=row.patient_name,
                    patient_age=row.patient_age,
                    transcript=row.transcript,
                    handoff_reason=row.handoff_reason,
                    handoff_summary=row.handoff_summary,
                    status=f"ended_{CallStatus}",
                )
        finally:
            db.close()
    return {"ok": "true"}


@router.post("/voice/handoff")
async def handoff_dial(
    CallSid: str = Form(...),
) -> Response:
    """
    TwiML for cold transfer: dial the configured human agent number.

    Twilio hits this after the worker redirects the live CallSid away from LiveKit SIP.
    """
    settings = get_settings()
    agent = (settings.twilio_human_agent_number or "").strip()
    log.info("Handoff dial TwiML | CallSid=%s agent=%s", CallSid, agent)

    db = SessionLocal()
    try:
        row = repo.get_by_call_sid(db, CallSid)
        summary = (row.handoff_summary if row else None) or ""
        reason = (row.handoff_reason if row else None) or ""
        if row:
            repo.update_intake(
                db,
                call_sid=CallSid,
                patient_name=row.patient_name,
                patient_age=row.patient_age,
                ready_to_proceed=row.ready_to_proceed,
                diseases=row.diseases,
                medications=row.medications,
                transcript=row.transcript,
                handoff_reason=row.handoff_reason,
                handoff_summary=row.handoff_summary,
                status="handed_off",
            )
    finally:
        db.close()

    response = VoiceResponse()
    if not _E164.match(agent):
        log.error("TWILIO_HUMAN_AGENT_NUMBER missing/invalid — cannot hand off")
        response.say(
            "I'm sorry, no human agent is available right now. Please try again later."
        )
        response.hangup()
        return Response(content=str(response), media_type="application/xml")

    # Whisper a short context line to the agent after they answer (caller hears hold).
    whisper_bits = [p for p in (reason, summary) if p]
    whisper = "Incoming handoff from AI intake. " + " ".join(whisper_bits)
    if len(whisper) > 400:
        whisper = whisper[:397] + "..."

    response.say("Please hold while I connect you.")
    dial: Dial = Dial(
        caller_id=settings.twilio_phone_number or None,
        answer_on_bridge=True,
    )
    # Number noun url: TwiML runs for the agent only before bridging.
    # Pass original CallSid so whisper can load the intake summary.
    whisper_url = (
        f"{settings.public_base_url.strip().rstrip('/')}/twilio/voice/handoff-whisper"
        f"?original_call_sid={CallSid}"
    )
    dial.number(agent, url=whisper_url)
    response.append(dial)
    log.debug("Handoff TwiML | CallSid=%s whisper_len=%s", CallSid, len(whisper))
    _HANDOFF_WHISPER[CallSid] = whisper
    return Response(content=str(response), media_type="application/xml")


@router.post("/voice/handoff-whisper")
async def handoff_whisper(
    original_call_sid: str = Query(default=""),
    CallSid: str = Form(default=""),
    ParentCallSid: str = Form(default=""),
) -> Response:
    """Say intake context to the human agent only (Dial number callback)."""
    key = original_call_sid or ParentCallSid or CallSid
    text = _HANDOFF_WHISPER.pop(key, None) if key else None
    if not text and key:
        db = SessionLocal()
        try:
            row = repo.get_by_call_sid(db, key)
            if row and row.handoff_summary:
                text = f"Incoming handoff. {row.handoff_summary}"
        finally:
            db.close()
    if not text:
        text = "Incoming handoff from the AI intake assistant."
    log.info("Handoff whisper | key=%s text=%r", key, text[:200])
    response = VoiceResponse()
    response.say(text)
    return Response(content=str(response), media_type="application/xml")
