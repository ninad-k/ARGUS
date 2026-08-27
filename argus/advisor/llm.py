"""Pluggable LLM backend for the AI advisor.

Ported from DRUVA's ``app/core/advisor/llm.py`` — same ``LLMBackend`` Protocol,
the same five backend implementations on shared ``httpx``, and the same
battle-tested ``parse_llm_json()`` brace-balancing extractor for local models
that wrap JSON in prose.

Supported providers:
- ``anthropic``         — Claude API (messages)
- ``openai``            — OpenAI API (chat.completions)
- ``openai_compatible`` — any OpenAI-compatible /v1/chat/completions server (e.g. vLLM, LM Studio)
- ``ollama``            — local Ollama at /api/chat (default: gemma3:4b)
- ``none``              — no LLM layer; advisor runs rules-only

ARGUS is single-user, so unlike DRUVA there is no per-user DB-backed config
lookup: the backend is built once from ``LLMSettings`` via ``build_backend()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import structlog

from argus.config.llm import LLMSettings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    raw: dict[str, Any] | None = None


class LLMBackend(Protocol):
    provider: str
    model: str

    async def complete(self, req: LLMRequest) -> LLMResponse: ...

    async def aclose(self) -> None:
        """Release any resources (e.g. an owned ``httpx.AsyncClient``).

        A backend built from an injected ``http`` client (see
        ``build_backend``) must NOT close it here -- the caller that
        injected it owns its lifecycle.
        """
        ...


class NoOpBackend:
    provider = "none"
    model = "-"

    async def complete(self, req: LLMRequest) -> LLMResponse:
        return LLMResponse(text="", provider=self.provider, model=self.model)

    async def aclose(self) -> None:
        return None


class AnthropicBackend:
    provider = "anthropic"

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        model: str,
        api_key: str,
        timeout_s: int = 120,
        owns_client: bool = False,
    ):
        self.http = http
        self.model = model
        self._api_key = api_key
        self._timeout = timeout_s
        self._owns_client = owns_client

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def complete(self, req: LLMRequest) -> LLMResponse:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
        }
        resp = await self.http.post(url, headers=headers, json=body, timeout=self._timeout)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        return LLMResponse(text=text, provider=self.provider, model=self.model, raw=data)


class OpenAIBackend:
    provider = "openai"

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: int = 120,
        owns_client: bool = False,
    ):
        self.http = http
        self.model = model
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._owns_client = owns_client

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def complete(self, req: LLMRequest) -> LLMResponse:
        url = f"{self._base}/chat/completions"
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        }
        resp = await self.http.post(url, headers=headers, json=body, timeout=self._timeout)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, provider=self.provider, model=self.model, raw=data)


class OpenAICompatibleBackend(OpenAIBackend):
    """OpenAI-compatible endpoints (vLLM, LM Studio, LiteLLM, etc.)."""

    provider = "openai_compatible"


class OllamaBackend:
    """Local Ollama server (https://github.com/ollama/ollama).

    Works great with Gemma: pull once with ``ollama pull gemma3:4b`` (the
    ARGUS default model).
    """

    provider = "ollama"

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: int = 120,
        owns_client: bool = False,
    ):
        self.http = http
        self.model = model
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._owns_client = owns_client

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http.aclose()

    async def complete(self, req: LLMRequest) -> LLMResponse:
        url = f"{self._base}/api/chat"
        body = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        }
        resp = await self.http.post(url, json=body, timeout=self._timeout)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        text = data.get("message", {}).get("content", "")
        return LLMResponse(text=text, provider=self.provider, model=self.model, raw=data)


def build_backend(settings: LLMSettings, *, http: httpx.AsyncClient | None = None) -> LLMBackend:
    """Build the configured ``LLMBackend`` from ``LLMSettings``.

    Unknown or disabled providers fall back to ``NoOpBackend`` rather than
    raising (DRUVA raises ``ValueError`` for an unknown provider, but ARGUS's
    pipeline must never fail because of LLM configuration — see
    ``pick_reviewer.review_picks``). A missing API key for a cloud provider
    also falls back to ``NoOpBackend`` with a warning, matching DRUVA.

    Callers should ``await backend.aclose()`` when done with a backend built
    here, to release an owned ``httpx.AsyncClient``; it's a no-op if ``http``
    was injected. See ``argus.pipeline._review_with_llm``.

    ``http`` is exposed mainly for tests (pass a client wired to
    ``httpx.MockTransport``); production callers can omit it and let each
    backend own a fresh ``httpx.AsyncClient``.
    """
    if not settings.enabled:
        return NoOpBackend()

    api_key = settings.api_key.get_secret_value() if settings.api_key else None
    provider = settings.provider

    if provider == "anthropic" and not api_key:
        logger.warning("advisor.llm.anthropic_missing_key")
        return NoOpBackend()
    if provider == "openai" and not api_key:
        logger.warning("advisor.llm.openai_missing_key")
        return NoOpBackend()
    if provider not in ("anthropic", "openai", "openai_compatible", "ollama"):
        if provider != "none":
            logger.warning("advisor.llm.unknown_provider", provider=provider)
        return NoOpBackend()

    # Only reached for a provider that actually needs an HTTP client -- avoids
    # constructing (and leaking) an unused ``httpx.AsyncClient`` for the
    # disabled/unknown/missing-key cases handled above.
    #
    # ``owns_client`` tracks whether *we* created the client here (production
    # path) vs. it being injected (tests, wired to a ``MockTransport``): only
    # a client we created ourselves gets closed by the backend's ``aclose()``
    # -- closing an injected client would be surprising for a caller that
    # still owns it.
    owns_client = http is None
    client = http if http is not None else httpx.AsyncClient()

    if provider == "anthropic":
        assert api_key is not None  # narrowed above
        return AnthropicBackend(
            http=client,
            model=settings.model,
            api_key=api_key,
            timeout_s=settings.timeout_seconds,
            owns_client=owns_client,
        )
    if provider == "openai":
        assert api_key is not None  # narrowed above
        return OpenAIBackend(
            http=client,
            model=settings.model,
            api_key=api_key,
            timeout_s=settings.timeout_seconds,
            owns_client=owns_client,
        )
    if provider == "openai_compatible":
        return OpenAICompatibleBackend(
            http=client,
            model=settings.model,
            api_key=api_key or "local",
            base_url=settings.base_url or "http://localhost:8000/v1",
            timeout_s=settings.timeout_seconds,
            owns_client=owns_client,
        )
    return OllamaBackend(
        http=client,
        model=settings.model,
        base_url=settings.base_url or "http://localhost:11434",
        timeout_s=settings.timeout_seconds,
        owns_client=owns_client,
    )


def parse_llm_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from an LLM response.

    Local models often wrap JSON in markdown or prose; we scan for the first
    balanced ``{...}`` block.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed: dict[str, Any] = json.loads(text[start : i + 1])
                    return parsed
                except json.JSONDecodeError:
                    return None
    return None
