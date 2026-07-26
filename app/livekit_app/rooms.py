"""LiveKit room + access token helpers."""

from __future__ import annotations

import uuid

from livekit import api

from app.config import get_settings
from app.logging_setup import get_logger

log = get_logger(__name__)


def make_room_name(call_sid: str) -> str:
    # LiveKit room names should be URL/SIP friendly.
    safe = "".join(ch if ch.isalnum() else "-" for ch in call_sid)
    room = f"intake-{safe[-24:]}-{uuid.uuid4().hex[:6]}"
    log.debug("Generated room_name=%s from call_sid=%s", room, call_sid)
    return room


async def create_room(room_name: str) -> None:
    settings = get_settings()
    log.debug("Creating LiveKit room | name=%s url=%s", room_name, settings.livekit_url)
    lk = api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    try:
        await lk.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=300,
                max_participants=4,
            )
        )
        log.info("LiveKit room created | name=%s", room_name)
    finally:
        await lk.aclose()


async def create_room_with_agent(room_name: str, *, metadata: str) -> None:
    """Create room and auto-dispatch healthcare-intake agent into it."""
    settings = get_settings()
    log.debug(
        "Creating LiveKit room+agent | name=%s url=%s metadata=%s",
        room_name,
        settings.livekit_url,
        metadata,
    )
    lk = api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    try:
        await lk.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=300,
                max_participants=4,
                agents=[
                    api.RoomAgentDispatch(
                        agent_name="healthcare-intake",
                        metadata=metadata,
                    )
                ],
            )
        )
        log.info("LiveKit room+agent created | name=%s", room_name)
    finally:
        await lk.aclose()


def create_participant_token(*, room_name: str, identity: str, name: str) -> str:
    settings = get_settings()
    log.debug(
        "Minting LiveKit token | room=%s identity=%s name=%s",
        room_name,
        identity,
        name,
    )
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    jwt = token.to_jwt()
    log.debug("LiveKit token minted | identity=%s len=%s", identity, len(jwt))
    return jwt


async def dispatch_agent(room_name: str, metadata: str) -> None:
    """Ask LiveKit to start our agent worker in this room."""
    settings = get_settings()
    log.debug("Dispatching agent | room=%s metadata=%s", room_name, metadata)
    lk = api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    try:
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="healthcare-intake",
                room=room_name,
                metadata=metadata,
            )
        )
        log.info("Agent dispatch created | room=%s agent=healthcare-intake", room_name)
    finally:
        await lk.aclose()
