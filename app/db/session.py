"""PostgreSQL engine + session factory."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base
from app.logging_setup import get_logger

log = get_logger(__name__)


def _normalize_url(url: str) -> str:
    """SQLAlchemy 2.x dropped the bare ``postgres://`` alias; map it to the psycopg driver."""
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


settings = get_settings()
database_url = _normalize_url(settings.database_url)

if not database_url.startswith("postgresql+psycopg://"):
    raise RuntimeError(
        f"DATABASE_URL must be a PostgreSQL URL, got: {settings.database_url!r}"
    )

engine = create_engine(
    database_url,
    echo=settings.log_level.upper() == "DEBUG",
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    # Managed Postgres drops idle connections; validate and rotate before use.
    pool_pre_ping=True,
    pool_recycle=1800,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    log.debug("Creating DB tables if missing")
    Base.metadata.create_all(bind=engine)
    _seed_clinic_settings()
    log.info("DB ready")


def _seed_clinic_settings() -> None:
    from app.db.clinic_repo import ensure_clinic_settings

    db = SessionLocal()
    try:
        ensure_clinic_settings(db, default_name=settings.clinic_name or None)
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
