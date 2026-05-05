from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx

from backend.app.config import Settings
from backend.app.utils.tokens import estimate_cost, estimate_tokens


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    mode: str
    warnings: list[str] = field(default_factory=list)


class LLMProvider(Protocol):
    model: str
    mode: str

    async def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        ...


class MockLLMProvider:
    model = "dev-deterministic"
    mode = "dev"

    async def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        question = prompt.split("Question:", 1)[-1].splitlines()[0].strip() if "Question:" in prompt else "the query"
        evidence = " ".join(
            line.strip(" -") for line in prompt.splitlines() if line.strip().startswith("[")
        )
        text = (
            f"Dev-mode answer for {question}. "
            f"Relevant evidence: {evidence[:700] if evidence else 'configure a live LLM for final generation.'}"
        )
        return LLMResponse(
            text=text,
            prompt_tokens=estimate_tokens((system or "") + prompt),
            completion_tokens=estimate_tokens(text),
            model=self.model,
            mode=self.mode,
            warnings=["No live LLM configured; deterministic dev provider used."],
        )


class OpenAICompatibleProvider:
    mode = "live"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.llm_model

    async def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.llm_api_base.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": self.settings.llm_model,
                    "messages": messages,
                    "temperature": self.settings.llm_temperature,
                },
            )
            response.raise_for_status()
            payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or estimate_tokens(str(messages))),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
            model=self.model,
            mode=self.mode,
            warnings=[] if usage else ["Provider did not return usage; token counts estimated."],
        )


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider in {"openai", "openai_compatible"} and settings.llm_api_key:
        return OpenAICompatibleProvider(settings)
    return MockLLMProvider()

