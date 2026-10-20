from __future__ import annotations

from typing import Optional, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from apps.insurance_claims.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


def build_chat_model(
    settings: Optional[Settings] = None,
    *,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> ChatOpenAI:
    cfg = settings or get_settings()
    if not cfg.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(
        model=model or cfg.openrouter_model,
        temperature=temperature,
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
    )


def structured_llm(
    schema: Type[T],
    settings: Optional[Settings] = None,
    *,
    temperature: float = 0.0,
    model: Optional[str] = None,
):
    """Return a model bound to emit `schema` via LangChain structured output."""
    llm = build_chat_model(settings, temperature=temperature, model=model)
    return llm.with_structured_output(schema)
