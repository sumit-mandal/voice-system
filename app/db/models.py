"""SQLAlchemy models for shared voice + email continuity."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ClinicSettings(Base):
    """Singleton clinic config (row id=1). Name is used in Ava prompts/greetings."""

    __tablename__ = "clinic_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class User(Base):
    """Channel-neutral family/caller identity (UUID primary key)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    preferred_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    verification_status: Mapped[str] = mapped_column(
        String(32), default="unverified", index=True
    )
    verification_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Consent flags (channel-specific permissions)
    consent_phone: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    consent_voicemail: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    consent_sms: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    consent_email: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    messages_may_name_clinic: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ChannelIdentity(Base):
    """Maps normalized phone/email → user UUID."""

    __tablename__ = "channel_identities"
    __table_args__ = (
        UniqueConstraint("channel", "value", name="uq_channel_identity_value"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)  # phone | email
    value: Mapped[str] = mapped_column(String(256), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class IntakeProfile(Base):
    """Channel-neutral Ava intake capture for a user."""

    __tablename__ = "intake_profiles"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Indexed key fields for lookup / routing
    caller_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    relationship_to_child: Mapped[str | None] = mapped_column(String(64), nullable=True)
    callback_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    child_first_name: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    child_last_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    child_dob: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    child_age_computed: Mapped[str | None] = mapped_column(String(64), nullable=True)
    home_city: Mapped[str | None] = mapped_column(String(64), nullable=True)
    home_zip: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    diagnosis_stated: Mapped[str | None] = mapped_column(Text, nullable=True)
    asd_diagnosis: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    insurance_carrier: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    primary_disposition: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    intake_complete: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # Full Ava capture blob (clinical, coverage, logistics, tasks, etc.)
    capture_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class Interaction(Base):
    """Channel-neutral interaction summary (not a raw transcript dump)."""

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    channel: Mapped[str] = mapped_column(String(16), index=True)  # phone | email
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    outcome: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="unverified")
    capture_delta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class CallSession(Base):
    __tablename__ = "call_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_sid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    room_name: Mapped[str] = mapped_column(String(128), index=True)
    caller_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
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
