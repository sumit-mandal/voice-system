"""Gemini LLM client via LangChain (langchain-google-genai)."""

from __future__ import annotations

import time
from collections.abc import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.latency import log_latency
from app.logging_setup import get_logger

log = get_logger(__name__)

_LLM: dict[tuple[str, bool], ChatGoogleGenerativeAI] = {}


def build_llm(*, json_mode: bool = False) -> ChatGoogleGenerativeAI:
    get_settings.cache_clear()
    settings = get_settings()

    cache_key = (settings.gemini_model, json_mode)
    cached = _LLM.get(cache_key)
    if cached is not None:
        return cached

    log.debug(
        "Building Gemini | model=%s temp=%s top_p=%s max_tokens=%s json_mode=%s",
        settings.gemini_model,
        settings.gemini_temperature,
        settings.gemini_top_p,
        settings.gemini_max_tokens,
        json_mode,
    )
    kwargs: dict[str, object] = {
        "model": settings.gemini_model,
        "google_api_key": settings.gemini_api_key,
        "temperature": settings.gemini_temperature,
        "top_p": settings.gemini_top_p,
        "max_output_tokens": settings.gemini_max_tokens,
    }
    # Thinking tokens come out of the same budget and truncate long JSON replies.
    optional = {"thinking_budget": settings.gemini_thinking_budget}
    if json_mode:
        optional["response_mime_type"] = "application/json"

    try:
        client = ChatGoogleGenerativeAI(**kwargs, **optional)
    except (TypeError, ValueError):
        log.warning(
            "Gemini client rejected optional args %s — falling back",
            sorted(optional),
            exc_info=True,
        )
        client = ChatGoogleGenerativeAI(**kwargs)

    _LLM[cache_key] = client
    log.info("Gemini client ready | model=%s json_mode=%s", settings.gemini_model, json_mode)
    return client


def _to_lc_messages(messages: list[dict[str, str]]) -> list:
    lc_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def chat_completion(messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
    """Run a chat completion and return assistant text."""
    settings = get_settings()
    llm = build_llm(json_mode=json_mode)
    log.debug(
        "chat_completion (Gemini) | model=%s messages=%s json_mode=%s",
        settings.gemini_model,
        len(messages),
        json_mode,
    )

    t0 = time.perf_counter()
    response = llm.invoke(_to_lc_messages(messages))
    log_latency("llm", (time.perf_counter() - t0) * 1000.0, model=settings.gemini_model)
    content = response.content if isinstance(response.content, str) else str(response.content)
    finish_reason = (response.response_metadata or {}).get("finish_reason")
    if finish_reason and str(finish_reason).upper() not in {"STOP", "1"}:
        log.warning(
            "Gemini stopped early | finish_reason=%s chars=%s max_tokens=%s",
            finish_reason,
            len(content),
            settings.gemini_max_tokens,
        )
    log.debug("Gemini content | %r", content[:1200])
    if not content.strip():
        raise ValueError("Gemini returned empty content")
    return content


# def chat_completion_stream(messages: list[dict[str, str]]) -> Iterator[str]:
#     """Yield text deltas from Gemini as they arrive (for low perceived latency)."""
#     settings = get_settings()
#     llm = build_llm()
#     log.debug(
#         "chat_completion_stream (Gemini) | model=%s messages=%s",
#         settings.gemini_model,
#         len(messages),
#     )
#     t0 = time.perf_counter()
#     first = True
#     for chunk in llm.stream(_to_lc_messages(messages)):
#         if first:
#             log_latency(
#                 "llm_ttft",
#                 (time.perf_counter() - t0) * 1000.0,
#                 model=settings.gemini_model,
#             )
#             first = False
#         content = chunk.content
#         if content is None:
#             continue
#         if not isinstance(content, str):
#             content = str(content)
#         if content:
#             yield content
#     log_latency("llm_stream_total", (time.perf_counter() - t0) * 1000.0, model=settings.gemini_model)
