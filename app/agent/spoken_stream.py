"""Fast streaming spoken replies (plain text) for low perceived chat latency."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from app.agent.llm import chat_completion_stream
from app.db.clinic_repo import get_clinic_name
from app.logging_setup import get_logger

log = get_logger(__name__)

CHAT_FILLERS = ("Okay…", "Got it…", "Sure…", "Alright…")


def stream_spoken_reply(
    *,
    user_text: str,
    pending_field: str | None,
    state_snapshot: dict[str, Any],
    history: list[dict[str, str]],
) -> Iterator[str]:
    """
    Stream a short plain-text Ava reply (no JSON) while full intake can run in parallel.
    """
    clinic = get_clinic_name()
    history_lines = []
    for turn in history[-6:]:
        history_lines.append(f"{turn['role'].upper()}: {turn['content']}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    system = f"""You are Ava, intake coordinator for {clinic}, chatting with a family over text.
Reply with ONE short natural message only — no JSON, no markdown, no bullet lists.
Ask at most one question. Keep it under 2 sentences.
If pending_field is set, advance that topic.
Never invent clinical guarantees or coverage quotes."""

    user = f"""Known slots:
{json.dumps(state_snapshot, indent=2, default=str)}

pending_field: {pending_field!r}

Recent conversation:
{history_block}

Latest user message:
{user_text!r}

Speak the next Ava reply now (plain text only)."""

    log.debug("stream_spoken_reply | pending=%r user=%r", pending_field, user_text[120])
    for delta in chat_completion_stream(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    ):
        yield delta
