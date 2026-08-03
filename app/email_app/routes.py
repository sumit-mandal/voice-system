"""HTTP routes for email agent — debug simulate + SES/SNS/S3 inbound."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote_plus

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.email_app.inbound_s3 import list_recent_keys, parse_raw_mime, parse_s3_email
from app.email_app.service import process_and_reply
from app.logging_setup import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/email", tags=["email"])


class DebugEmailRequest(BaseModel):
    from_address: str = Field(
        ...,
        description="Sender (also SES reply To). Must be SES-verified in sandbox.",
        examples=["sumit.mandal13100@gmail.com"],
    )
    subject: str = Field(default="My history records")
    body_text: str = Field(
        default="Please tell me about my history records.",
        min_length=1,
    )
    send_reply: bool = Field(
        default=True,
        description="If true, send auto-reply via SES to from_address",
    )
    patient_id: str | None = None


class ProcessS3Request(BaseModel):
    key: str = Field(..., description="S3 object key written by SES receipt rule")
    bucket: str | None = Field(
        default=None,
        description="Defaults to SES_INBOUND_BUCKET",
    )
    send_reply: bool = True


@router.post("/debug")
def debug_email(body: DebugEmailRequest) -> dict[str, Any]:
    """
    Sandbox-friendly: pretend we received an email, run LangGraph, send SES reply.

    Both SES_FROM_ADDRESS and from_address must be verified while in the SES sandbox.
    """
    log.info(
        "POST /email/debug | from=%s subject=%r send_reply=%s",
        body.from_address,
        body.subject,
        body.send_reply,
    )
    try:
        out = process_and_reply(
            from_address=body.from_address,
            subject=body.subject,
            body_text=body.body_text,
            send_reply=body.send_reply,
            patient_id=body.patient_id,
        )
    except Exception as exc:
        log.exception("/email/debug failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if body.send_reply and out.get("send_error"):
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Agent drafted a reply but SES send failed (sandbox?)",
                "send_error": out["send_error"],
                "draft": {
                    "reply_subject": out.get("reply_subject"),
                    "reply_body": out.get("reply_body"),
                    "intent": out.get("intent"),
                },
            },
        )
    return out


@router.get("/ses/inbound")
def list_inbound(max_keys: int = 20) -> dict[str, Any]:
    """List recent objects in SES_INBOUND_BUCKET (proves MX/receipt rule is working)."""
    settings = get_settings()
    bucket = (settings.ses_inbound_bucket or "").strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="SES_INBOUND_BUCKET not set")
    try:
        keys = list_recent_keys(bucket=bucket, max_keys=max_keys)
    except Exception as exc:
        log.exception("list inbound S3 failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"bucket": bucket, "count": len(keys), "objects": keys}


@router.post("/ses/process-s3")
def process_s3(body: ProcessS3Request) -> dict[str, Any]:
    """Parse one raw MIME object from S3 and send auto-reply via SES."""
    settings = get_settings()
    bucket = (body.bucket or settings.ses_inbound_bucket or "").strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="SES_INBOUND_BUCKET not set")
    try:
        parsed = parse_s3_email(bucket=bucket, key=body.key)
    except Exception as exc:
        log.exception("parse S3 email failed | key=%s", body.key)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not parsed.get("from_address"):
        raise HTTPException(status_code=400, detail="Could not parse From address from MIME")

    out = process_and_reply(
        from_address=parsed["from_address"],
        subject=parsed["subject"],
        body_text=parsed["body_text"],
        message_id=parsed.get("message_id") or body.key,
        send_reply=body.send_reply,
    )
    out["s3_bucket"] = bucket
    out["s3_key"] = body.key
    if body.send_reply and out.get("send_error"):
        raise HTTPException(status_code=502, detail=out)
    return out


@router.post("/ses/process-latest")
def process_latest(send_reply: bool = True) -> dict[str, Any]:
    """
    Process the newest object in SES_INBOUND_BUCKET.

    Use after emailing intake@… to confirm mail landed in S3, then auto-reply.
    """
    settings = get_settings()
    bucket = (settings.ses_inbound_bucket or "").strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="SES_INBOUND_BUCKET not set")
    try:
        keys = list_recent_keys(bucket=bucket, max_keys=5)
    except Exception as exc:
        log.exception("list S3 for process-latest failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not keys:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No objects in s3://{bucket}/ — SES is not receiving mail yet. "
                "Check MX for sumitmandal.in points to inbound-smtp.<region>.amazonaws.com "
                "and an active receipt rule writes to this bucket."
            ),
        )
    latest = keys[0]["key"]
    log.info("process-latest | key=%s", latest)
    return process_s3(ProcessS3Request(key=latest, bucket=bucket, send_reply=send_reply))


@router.post("/ses/sns")
async def ses_sns_webhook(request: Request) -> dict[str, str]:
    """
    SNS endpoint for SES Received notifications and/or S3 event notifications.

    HTTPS subscription:
      POST https://<PUBLIC_BASE_URL>/email/ses/sns
    """
    raw = await request.body()
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON") from exc

    msg_type = (
        request.headers.get("x-amz-sns-message-type")
        or envelope.get("Type")
        or ""
    )
    log.info("SES SNS | Type=%s", msg_type)

    if msg_type == "SubscriptionConfirmation":
        subscribe_url = envelope.get("SubscribeURL")
        if not subscribe_url:
            raise HTTPException(status_code=400, detail="missing SubscribeURL")
        log.info("Confirming SNS subscription | url=%s", subscribe_url[:120])
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(subscribe_url)
            resp.raise_for_status()
        return {"ok": "subscribed"}

    if msg_type == "UnsubscribeConfirmation":
        return {"ok": "unsubscribed"}

    if msg_type != "Notification":
        log.warning("Ignoring SNS message type=%s", msg_type)
        return {"ok": "ignored"}

    try:
        notification = json.loads(envelope["Message"])
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="invalid SNS Message") from exc

    # S3 event notification (bucket → SNS)
    if "Records" in notification:
        processed = 0
        for rec in notification.get("Records") or []:
            if rec.get("eventSource") != "aws:s3":
                continue
            bucket = rec.get("s3", {}).get("bucket", {}).get("name")
            key = rec.get("s3", {}).get("object", {}).get("key")
            if not bucket or not key:
                continue
            key = unquote_plus(key)
            log.info("S3 event inbound | s3://%s/%s", bucket, key)
            parsed = parse_s3_email(bucket=bucket, key=key)
            process_and_reply(
                from_address=parsed["from_address"],
                subject=parsed["subject"],
                body_text=parsed["body_text"],
                message_id=parsed.get("message_id") or key,
                send_reply=True,
            )
            processed += 1
        return {"ok": "processed", "count": str(processed)}

    notification_type = notification.get("notificationType") or notification.get("eventType")
    if notification_type and notification_type != "Received":
        log.info("Ignoring SES notificationType=%s", notification_type)
        return {"ok": "ignored"}

    parsed = _extract_inbound_ses(notification)
    if not parsed:
        log.warning("Could not extract inbound email from SES notification")
        return {"ok": "no_content"}

    log.info(
        "Inbound SES mail | from=%s subject=%r message_id=%s",
        parsed["from_address"],
        parsed["subject"],
        parsed["message_id"],
    )
    process_and_reply(
        from_address=parsed["from_address"],
        subject=parsed["subject"],
        body_text=parsed["body_text"],
        message_id=parsed["message_id"],
        send_reply=True,
    )
    return {"ok": "processed"}


def _extract_inbound_ses(notification: dict[str, Any]) -> dict[str, str] | None:
    """Pull from/subject/body/message_id from SES Received notification (+ optional S3)."""
    mail = notification.get("mail") or {}
    common = mail.get("commonHeaders") or {}
    message_id = (mail.get("messageId") or common.get("messageId") or "").strip()
    subject = ""
    if isinstance(common.get("subject"), str):
        subject = common["subject"]
    elif isinstance(mail.get("headers"), list):
        for h in mail["headers"]:
            if str(h.get("name", "")).lower() == "subject":
                subject = str(h.get("value") or "")
                break

    from_list = common.get("from") or mail.get("source") or []
    if isinstance(from_list, list) and from_list:
        from_address = str(from_list[0])
    elif isinstance(from_list, str):
        from_address = from_list
    else:
        from_address = str(mail.get("source") or "")

    body_text = ""
    content = notification.get("content")
    if isinstance(content, str) and content.strip():
        body_text = parse_raw_mime(content.encode("utf-8", errors="replace"))["body_text"]

    receipt = notification.get("receipt") or {}
    action = receipt.get("action") or {}
    settings = get_settings()
    if action.get("type") == "S3":
        bucket = action.get("bucketName") or settings.ses_inbound_bucket
        key = action.get("objectKey")
        if bucket and key:
            parsed = parse_s3_email(bucket=bucket, key=key)
            return {
                "from_address": from_address or parsed["from_address"],
                "subject": subject or parsed["subject"],
                "body_text": parsed["body_text"],
                "message_id": message_id or parsed["message_id"],
            }

    if not body_text and settings.ses_inbound_bucket and message_id:
        try:
            parsed = parse_s3_email(bucket=settings.ses_inbound_bucket, key=message_id)
            return {
                "from_address": from_address or parsed["from_address"],
                "subject": subject or parsed["subject"],
                "body_text": parsed["body_text"],
                "message_id": message_id or parsed["message_id"],
            }
        except Exception:
            log.exception("Fallback S3 fetch failed | key=%s", message_id)

    if not from_address:
        return None
    if not body_text:
        body_text = "(empty body — check SES receipt rule includes S3 or SNS content)"

    return {
        "from_address": from_address,
        "subject": subject or "(no subject)",
        "body_text": body_text,
        "message_id": message_id or f"ses-{from_address}",
    }


@router.get("/health")
def email_health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "ses_from": settings.ses_from_address or "",
        "aws_region": settings.aws_region,
        "ses_inbound_bucket": settings.ses_inbound_bucket or "",
    }
