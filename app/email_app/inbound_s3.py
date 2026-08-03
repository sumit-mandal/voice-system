"""Load inbound emails from the SES receipt-rule S3 bucket."""

from __future__ import annotations

from email import message_from_bytes
from email.policy import default
from email.utils import parseaddr
from typing import Any

import boto3

from app.config import get_settings
from app.logging_setup import get_logger

log = get_logger(__name__)


def _s3_client():
    settings = get_settings()
    kwargs: dict = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def parse_raw_mime(raw: bytes) -> dict[str, str]:
    msg = message_from_bytes(raw, policy=default)
    from_raw = str(msg.get("From") or "")
    _, from_addr = parseaddr(from_raw)
    subject = str(msg.get("Subject") or "(no subject)")
    message_id = str(msg.get("Message-ID") or msg.get("Message-Id") or "").strip()

    body_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body_text = part.get_content()
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    body_text = (
                        payload.decode("utf-8", errors="replace")
                        if isinstance(payload, bytes)
                        else str(payload)
                    )
                break
    else:
        try:
            if msg.get_content_type() == "text/plain":
                body_text = msg.get_content()
            else:
                payload = msg.get_payload(decode=True) or b""
                body_text = (
                    payload.decode("utf-8", errors="replace")
                    if isinstance(payload, bytes)
                    else str(payload)
                )
        except Exception:
            payload = msg.get_payload(decode=True) or b""
            body_text = (
                payload.decode("utf-8", errors="replace")
                if isinstance(payload, bytes)
                else str(payload)
            )

    return {
        "from_address": (from_addr or from_raw).strip(),
        "subject": subject,
        "body_text": (body_text or "").strip() or "(empty body)",
        "message_id": message_id or "",
    }


def fetch_s3_object(*, bucket: str, key: str) -> bytes:
    log.info("S3 get_object | s3://%s/%s", bucket, key)
    return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()


def parse_s3_email(*, bucket: str, key: str) -> dict[str, str]:
    raw = fetch_s3_object(bucket=bucket, key=key)
    parsed = parse_raw_mime(raw)
    if not parsed.get("message_id"):
        parsed["message_id"] = key
    parsed["s3_bucket"] = bucket
    parsed["s3_key"] = key
    return parsed


def list_recent_keys(*, bucket: str, prefix: str = "", max_keys: int = 20) -> list[dict[str, Any]]:
    client = _s3_client()
    kwargs: dict[str, Any] = {"Bucket": bucket, "MaxKeys": max_keys}
    if prefix:
        kwargs["Prefix"] = prefix
    resp = client.list_objects_v2(**kwargs)
    contents = resp.get("Contents") or []
    contents = sorted(contents, key=lambda o: o.get("LastModified") or 0, reverse=True)
    out: list[dict[str, Any]] = []
    for obj in contents[:max_keys]:
        out.append(
            {
                "key": obj["Key"],
                "size": obj.get("Size"),
                "last_modified": obj["LastModified"].isoformat()
                if obj.get("LastModified")
                else None,
            }
        )
    return out
