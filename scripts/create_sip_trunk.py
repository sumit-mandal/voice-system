"""Create inbound SIP trunk + dispatch rule on self-hosted LiveKit for Twilio."""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from livekit import api

from app.logging_setup import get_logger, setup_logging

log = get_logger(__name__)


async def main() -> None:
    load_dotenv()
    setup_logging(os.getenv("LOG_LEVEL", "DEBUG"))

    url = os.environ["LIVEKIT_URL"]
    key = os.environ["LIVEKIT_API_KEY"]
    secret = os.environ["LIVEKIT_API_SECRET"]
    number = os.environ["TWILIO_PHONE_NUMBER"]
    existing_trunk = os.getenv("LIVEKIT_SIP_TRUNK_ID", "").strip()

    lk = api.LiveKitAPI(url, key, secret)
    try:
        trunk_id = existing_trunk
        if not trunk_id:
            # Prefer an existing trunk that already owns this number.
            listed = await lk.sip.list_sip_inbound_trunk(api.ListSIPInboundTrunkRequest())
            for t in listed.items or []:
                if number in (t.numbers or []):
                    trunk_id = t.sip_trunk_id
                    log.info("Found existing trunk for number | sip_trunk_id=%s", trunk_id)
                    break

        if not trunk_id:
            log.info("Creating inbound SIP trunk | url=%s number=%s", url, number)
            # numbers=[] so sip:{room}@host matches (Twilio Dial Sip uses room as To-user).
            # Restrict by Twilio signaling IP ranges instead.
            trunk = await lk.sip.create_sip_inbound_trunk(
                api.CreateSIPInboundTrunkRequest(
                    trunk=api.SIPInboundTrunkInfo(
                        name="twilio-inbound",
                        numbers=[],
                        allowed_addresses=[
                            "54.172.60.0/23",
                            "34.203.250.0/23",
                            "54.244.51.0/24",
                            "54.171.127.192/26",
                            "35.156.191.128/25",
                            "54.65.63.192/26",
                            "54.169.127.128/26",
                            "54.252.254.64/26",
                            "177.71.206.192/26",
                        ],
                        krisp_enabled=True,
                    )
                )
            )
            trunk_id = trunk.sip_trunk_id
            log.info("Created trunk | sip_trunk_id=%s", trunk_id)
        else:
            log.info("Reusing existing trunk | sip_trunk_id=%s", trunk_id)
            # Ensure trunk accepts room-name SIP dials from Twilio.
            await lk.sip.update_inbound_trunk(
                trunk_id,
                api.SIPInboundTrunkInfo(
                    name="twilio-inbound",
                    numbers=[],
                    allowed_addresses=[
                        "54.172.60.0/23",
                        "34.203.250.0/23",
                        "54.244.51.0/24",
                        "54.171.127.192/26",
                        "35.156.191.128/25",
                        "54.65.63.192/26",
                        "54.169.127.128/26",
                        "54.252.254.64/26",
                        "177.71.206.192/26",
                    ],
                    krisp_enabled=True,
                ),
            )
            log.info("Updated trunk to accept room SIP dials | sip_trunk_id=%s", trunk_id)

        # Map sip:{room}@host → LiveKit room named {room} (Twilio <Dial><Sip>).
        log.info("Creating callee dispatch rule for trunk=%s", trunk_id)
        rule = await lk.sip.create_sip_dispatch_rule(
            api.CreateSIPDispatchRuleRequest(
                name="twilio-to-room",
                trunk_ids=[trunk_id],
                rule=api.SIPDispatchRule(
                    dispatch_rule_callee=api.SIPDispatchRuleCallee(
                        room_prefix="",
                        randomize=False,
                    )
                ),
            )
        )
        log.info("Created dispatch rule | id=%s", rule.sip_dispatch_rule_id)

        print(f"\nLIVEKIT_SIP_TRUNK_ID={trunk_id}")
        print(f"LIVEKIT_SIP_DISPATCH_RULE_ID={rule.sip_dispatch_rule_id}\n")
        print("Paste LIVEKIT_SIP_TRUNK_ID into .env if it changed.")
    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
