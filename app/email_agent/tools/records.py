"""Fetch shared continuity records for email Ava."""

from __future__ import annotations

from typing import Any

from app.continuity.context import load_continuity_context
from app.db.session import SessionLocal
from app.logging_setup import get_logger

log = get_logger(__name__)


def fetch_history_records(
    *,
    patient_id: str | None,
    from_address: str,
) -> dict[str, Any]:
    """Load intake profile + recent interactions for a resolved user UUID."""
    user_id = patient_id
    db = SessionLocal()
    try:
        ctx = load_continuity_context(db, user_id)
        log.info(
            "fetch_history_records | user_id=%s from=%s verified=%s has_history=%s",
            user_id,
            from_address,
            ctx.get("verified"),
            ctx.get("has_history"),
        )
        if not user_id:
            return {
                "ok": False,
                "error": "patient_id_unresolved",
                "records": [],
                "continuity": ctx,
            }
        return {
            "ok": True,
            "patient_id": user_id,
            "user_id": user_id,
            "verified": ctx["verified"],
            "profile": ctx["profile"] if ctx["verified"] else {},
            "interactions": ctx["interactions"] if ctx["verified"] else [],
            "prompt_block": ctx["prompt_block"],
            # Keep a "records" key for older draft prompts
            "records": ctx["interactions"] if ctx["verified"] else [],
        }
    finally:
        db.close()
