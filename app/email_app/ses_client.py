"""SES send helper for email agent auto-replies."""

from __future__ import annotations

import re
from email.utils import parseaddr

import boto3
from botocore.exceptions import ClientError

from app.config import get_settings
from app.logging_setup import get_logger

log = get_logger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _ses_client():
    settings = get_settings()
    kwargs: dict = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("ses", **kwargs)


def normalize_address(raw: str) -> str:
    """Extract bare email from 'Name <addr@x.com>' or return stripped address."""
    _, addr = parseaddr(raw or "")
    addr = (addr or raw or "").strip()
    return addr


def send_auto_reply(
    *,
    to_address: str,
    subject: str,
    body_text: str,
    in_reply_to: str | None = None,
    references: str | None = None,
) -> str:
    """
    Send a plain-text reply via SES.

    Sandbox: both SES_FROM_ADDRESS and to_address must be verified identities
    (or you must leave the SES sandbox).
    Returns SES MessageId.
    """
    settings = get_settings()
    from_addr = (settings.ses_from_address or "").strip()
    to_addr = normalize_address(to_address)
    if not from_addr or not _EMAIL_RE.match(from_addr):
        raise ValueError("SES_FROM_ADDRESS must be a verified email, e.g. intake@sumitmandal.in")
    if not to_addr or not _EMAIL_RE.match(to_addr):
        raise ValueError(f"Invalid to_address: {to_address!r}")

    headers: dict[str, str] = {}
    if in_reply_to:
        headers["In-Reply-To"] = in_reply_to
    if references or in_reply_to:
        headers["References"] = references or in_reply_to or ""

    log.info(
        "SES send_auto_reply | from=%s to=%s subject=%r region=%s",
        from_addr,
        to_addr,
        subject,
        settings.aws_region,
    )

    client = _ses_client()
    try:
        # Simple send_email — fine for sandbox demos. Use send_raw_email if you
        # need custom headers beyond Reply-To; we pass In-Reply-To via raw below
        # when threading ids are present.
        if headers:
            raw = _build_raw_message(
                from_addr=from_addr,
                to_addr=to_addr,
                subject=subject,
                body_text=body_text,
                extra_headers=headers,
            )
            resp = client.send_raw_email(
                Source=from_addr,
                Destinations=[to_addr],
                RawMessage={"Data": raw},
            )
        else:
            resp = client.send_email(
                Source=from_addr,
                Destination={"ToAddresses": [to_addr]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {"Text": {"Data": body_text, "Charset": "UTF-8"}},
                },
            )
    except ClientError as exc:
        log.exception("SES send failed | to=%s", to_addr)
        raise RuntimeError(f"SES send failed: {exc}") from exc

    message_id = resp.get("MessageId") or ""
    log.info("SES send ok | MessageId=%s", message_id)
    return message_id


def _build_raw_message(
    *,
    from_addr: str,
    to_addr: str,
    subject: str,
    body_text: str,
    extra_headers: dict[str, str],
) -> bytes:
    lines = [
        f"From: {from_addr}",
        f"To: {to_addr}",
        f"Subject: {subject}",
        "MIME-Version: 1.0",
        'Content-Type: text/plain; charset="UTF-8"',
        "Content-Transfer-Encoding: 8bit",
    ]
    for key, value in extra_headers.items():
        if value:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append(body_text)
    return "\r\n".join(lines).encode("utf-8")
