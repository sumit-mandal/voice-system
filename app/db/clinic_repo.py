"""Clinic settings — name used across voice/email Ava surfaces."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import ClinicSettings
from app.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_CLINIC_ID = 1
DEFAULT_CLINIC_NAME = "Our Clinic"

_name_cache: str | None = None


def clear_clinic_name_cache() -> None:
    global _name_cache
    _name_cache = None


def ensure_clinic_settings(
    db: Session,
    *,
    default_name: str | None = None,
) -> ClinicSettings:
    """Ensure singleton clinic_settings row exists; return it."""
    row = db.query(ClinicSettings).filter(ClinicSettings.id == DEFAULT_CLINIC_ID).one_or_none()
    if row is None:
        name = (default_name or DEFAULT_CLINIC_NAME).strip() or DEFAULT_CLINIC_NAME
        row = ClinicSettings(id=DEFAULT_CLINIC_ID, name=name)
        db.add(row)
        db.commit()
        db.refresh(row)
        log.info("Seeded clinic_settings | name=%r", row.name)
        clear_clinic_name_cache()
    return row


def get_clinic_name(db: Session | None = None) -> str:
    """Return clinic display name from DB (cached after first read)."""
    global _name_cache
    if _name_cache:
        return _name_cache

    owns_session = db is None
    if owns_session:
        from app.db.session import SessionLocal

        db = SessionLocal()
    assert db is not None
    try:
        row = ensure_clinic_settings(db)
        _name_cache = row.name.strip() or DEFAULT_CLINIC_NAME
        return _name_cache
    finally:
        if owns_session:
            db.close()


def set_clinic_name(db: Session, name: str) -> ClinicSettings:
    """Update clinic display name and refresh cache."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("clinic name must be non-empty")
    row = ensure_clinic_settings(db)
    row.name = cleaned
    db.commit()
    db.refresh(row)
    clear_clinic_name_cache()
    log.info("Updated clinic name | name=%r", row.name)
    return row
