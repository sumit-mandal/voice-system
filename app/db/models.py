"""SQLAlchemy models for healthcare intake calls."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CallSession(Base):
    __tablename__ = "call_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_sid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    room_name: Mapped[str] = mapped_column(String(128), index=True)
    caller_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="in_progress")
    patient_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    patient_age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ready_to_proceed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    diseases: Mapped[str | None] = mapped_column(Text, nullable=True)
    medications: Mapped[str | None] = mapped_column(Text, nullable=True)
    handoff_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    handoff_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
