#!/usr/bin/env python3
"""Smoke-test shared UUID continuity + verification gate (no LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.continuity.context import build_continuity_block, load_continuity_context
from app.db import continuity_repo as crepo
from app.db.session import SessionLocal, init_db


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        email = "parent.smoke@example.com"
        phone = "+13135550100"

        email_user = crepo.find_or_create_user_by_email(db, email)
        assert email_user is not None
        crepo.upsert_intake_profile(
            db,
            email_user.id,
            caller_name="Danielle",
            relationship_to_child="mom",
            child_first_name="Marcus",
            child_dob="2024-03-14",
            diagnosis_stated="ASD Level 2",
            asd_diagnosis=True,
            insurance_carrier="Meridian Medicaid",
            home_city="Ferndale",
            home_zip="48220",
            primary_disposition="Callback Queue",
            capture_delta={
                "speech_requested": True,
                "medicaid_pihp_status": "unknown",
                "scenario_id": 1,
            },
        )
        crepo.append_interaction(
            db,
            user_id=email_user.id,
            channel="email",
            summary=(
                "email interaction; child=Marcus; dx_stated=ASD Level 2; "
                "coverage=Meridian Medicaid; disposition=Callback Queue"
            ),
            external_id="smoke-email-1",
            outcome="Callback Queue",
            verification_status="unverified",
        )

        # Pre-verify gate: phone contact is a NEW user initially
        phone_user = crepo.find_or_create_user_by_phone(db, phone)
        assert phone_user is not None
        assert phone_user.id != email_user.id

        unverified_block = build_continuity_block(
            db, phone_user.id, verified=False
        )
        assert "Do NOT disclose" in unverified_block or "No prior interactions" in unverified_block
        # Phone user has no history yet
        assert "No prior interactions" in unverified_block

        # Email user's unverified block should hint history exists without leaking dx
        email_unverified = build_continuity_block(db, email_user.id, verified=False)
        assert "possible prior contact" in email_unverified.lower()
        assert "ASD Level 2" not in email_unverified
        assert "Meridian" not in email_unverified

        # Cross-channel merge via name+DOB (as voice/email agents do after verify)
        matched = crepo.find_user_by_caller_and_child_dob(
            db, caller_name="Danielle", child_dob="2024-03-14"
        )
        assert matched is not None
        assert matched.id == email_user.id

        merged = crepo.merge_channel_onto_user(
            db,
            from_user_id=phone_user.id,
            to_user_id=email_user.id,
            channel="phone",
            value=phone,
        )
        assert merged.id == email_user.id
        crepo.set_verification_status(
            db, email_user.id, status="verified", method="smoke_name_dob"
        )

        # Same phone now resolves to email UUID
        by_phone = crepo.find_user_by_phone(db, phone)
        assert by_phone is not None and by_phone.id == email_user.id

        verified_ctx = load_continuity_context(db, email_user.id)
        assert verified_ctx["verified"] is True
        assert verified_ctx["profile"]["child_first_name"] == "Marcus"
        assert verified_ctx["profile"]["diagnosis_stated"] == "ASD Level 2"
        assert any(i["channel"] == "email" for i in verified_ctx["interactions"])
        assert "ASD Level 2" in verified_ctx["prompt_block"]
        assert "Callback Queue" in verified_ctx["prompt_block"]

        # Simulate call→email: append phone interaction on same UUID
        crepo.append_interaction(
            db,
            user_id=email_user.id,
            channel="phone",
            summary=(
                "phone interaction; child=Marcus; disposition=Callback Queue; "
                "continued from email intake"
            ),
            external_id="smoke-call-1",
            outcome="Callback Queue",
            verification_status="verified",
        )
        by_email = crepo.find_user_by_email(db, email)
        assert by_email is not None and by_email.id == email_user.id
        interactions = crepo.get_recent_interactions(db, email_user.id, limit=5)
        channels = {i.channel for i in interactions}
        assert channels == {"email", "phone"}

        print("PASS: shared UUID continuity + verification gate")
        print(f"  user_id={email_user.id}")
        print(f"  email={email} phone={phone}")
        print(f"  interactions={len(interactions)} channels={sorted(channels)}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
