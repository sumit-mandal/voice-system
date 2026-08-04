"""SQLite engine + session factory."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base
from app.logging_setup import get_logger

log = get_logger(__name__)


def _ensure_sqlite_parent(url: str) -> None:
    if not url.startswith("sqlite:///"):
        return
    path = Path(url.removeprefix("sqlite:///"))
    if path.parent and str(path.parent) not in (".", ""):
        path.parent.mkdir(parents=True, exist_ok=True)
        log.debug("Ensured SQLite parent dir exists: %s", path.parent)


settings = get_settings()
_ensure_sqlite_parent(settings.database_url)

_IS_SQLITE = settings.database_url.startswith("sqlite:///")

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    echo=settings.log_level.upper() == "DEBUG",
)

if _IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        log.debug("SQLite connection opened; foreign_keys=ON")


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    log.debug("Creating DB tables if missing | url=%s", settings.database_url)
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite_columns()
    _seed_clinic_settings()
    log.info("DB ready")


def _seed_clinic_settings() -> None:
    from app.db.clinic_repo import ensure_clinic_settings

    db = SessionLocal()
    try:
        ensure_clinic_settings(db, default_name=settings.clinic_name or None)
    finally:
        db.close()


def _migrate_sqlite_columns() -> None:
    """Add new intake columns to existing SQLite DBs without wiping data."""
    if not settings.database_url.startswith("sqlite:///"):
        return
    needed = {
        "ready_to_proceed": "BOOLEAN",
        "diseases": "TEXT",
        "medications": "TEXT",
        "handoff_reason": "VARCHAR(256)",
        "handoff_summary": "TEXT",
        "user_id": "VARCHAR(36)",
    }
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(call_sessions)").fetchall()
        existing = {row[1] for row in rows}
        for col, col_type in needed.items():
            if col in existing:
                continue
            sql = f"ALTER TABLE call_sessions ADD COLUMN {col} {col_type}"
            log.info("Migrating SQLite | %s", sql)
            conn.exec_driver_sql(sql)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
