"""Gemini LLM client via LangChain (langchain-google-genai)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.logging_setup import get_logger

log = get_logger(__name__)

_LLM: ChatGoogleGenerativeAI | None = None


def build_llm() -> ChatGoogleGenerativeAI:
    global _LLM
    get_settings.cache_clear()
    settings = get_settings()

    if _LLM is not None:
        current = getattr(_LLM, "model", None) or getattr(_LLM, "model_name", None)
        if current == settings.gemini_model:
            log.debug("Reusing cached ChatGoogleGenerativeAI | model=%s", current)
            return _LLM
        log.info("Gemini model changed %s → %s — rebuilding client", current, settings.gemini_model)

    log.debug(
        "Building Gemini | model=%s temp=%s top_p=%s max_tokens=%s",
        settings.gemini_model,
        settings.gemini_temperature,
        settings.gemini_top_p,
        settings.gemini_max_tokens,
    )
    _LLM = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.gemini_temperature,
        top_p=settings.gemini_top_p,
        max_output_tokens=settings.gemini_max_tokens,
    )
    log.info("Gemini client ready | model=%s", settings.gemini_model)
    return _LLM


def chat_completion(messages: list[dict[str, str]]) -> str:
    """Run a chat completion and return assistant text."""
    settings = get_settings()
    llm = build_llm()
    log.debug(
        "chat_completion (Gemini) | model=%s messages=%s",
        settings.gemini_model,
        len(messages),
    )

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

    response = llm.invoke(lc_messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    log.debug("Gemini content | %r", content[:1200])
    if not content.strip():
        raise ValueError("Gemini returned empty content")
    return content
