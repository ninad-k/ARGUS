"""``parse_llm_json`` extraction cases, ``build_backend`` provider selection,
and the Ollama backend's request/response shape against a mocked transport.
"""

from __future__ import annotations

import json

import httpx
import pytest

from argus.advisor.llm import (
    LLMRequest,
    NoOpBackend,
    OllamaBackend,
    build_backend,
    parse_llm_json,
)
from argus.config.llm import LLMSettings

# --- parse_llm_json ----------------------------------------------------------


def test_parse_llm_json_clean_object() -> None:
    text = '{"symbol": "AAPL", "verdict": "buy"}'
    assert parse_llm_json(text) == {"symbol": "AAPL", "verdict": "buy"}


def test_parse_llm_json_wrapped_in_prose() -> None:
    text = 'Sure, here is my analysis:\n{"symbol": "AAPL", "verdict": "buy"}\nHope that helps!'
    assert parse_llm_json(text) == {"symbol": "AAPL", "verdict": "buy"}


def test_parse_llm_json_fenced_code_block() -> None:
    text = '```json\n{"symbol": "AAPL", "verdict": "buy"}\n```'
    assert parse_llm_json(text) == {"symbol": "AAPL", "verdict": "buy"}


def test_parse_llm_json_nested_braces_in_strings() -> None:
    text = '{"thesis": "a {test} case", "verdict": "watch"}'
    assert parse_llm_json(text) == {"thesis": "a {test} case", "verdict": "watch"}


def test_parse_llm_json_no_braces_returns_none() -> None:
    assert parse_llm_json("just plain prose, no JSON here") is None


def test_parse_llm_json_empty_string_returns_none() -> None:
    assert parse_llm_json("") is None


def test_parse_llm_json_unbalanced_braces_returns_none() -> None:
    assert parse_llm_json('{"symbol": "AAPL"') is None


def test_parse_llm_json_malformed_inside_braces_returns_none() -> None:
    # Balanced braces, but not valid JSON inside them.
    assert parse_llm_json("{not json at all}") is None


# --- build_backend -------------------------------------------------------------


async def test_build_backend_disabled_returns_noop() -> None:
    settings = LLMSettings(enabled=False, _env_file=None)  # type: ignore[call-arg]
    backend = build_backend(settings)
    assert isinstance(backend, NoOpBackend)
    resp = await backend.complete(LLMRequest(system="s", user="u"))
    assert resp.text == ""
    assert resp.provider == "none"
    assert resp.model == "-"


def test_build_backend_unknown_provider_returns_noop() -> None:
    settings = LLMSettings(provider="carrier_pigeon", _env_file=None)  # type: ignore[call-arg]
    backend = build_backend(settings)
    assert isinstance(backend, NoOpBackend)


def test_build_backend_anthropic_without_key_returns_noop() -> None:
    settings = LLMSettings(provider="anthropic", api_key=None, _env_file=None)  # type: ignore[call-arg]
    backend = build_backend(settings)
    assert isinstance(backend, NoOpBackend)


# --- OllamaBackend request construction ----------------------------------------


async def test_ollama_backend_request_and_response_shape() -> None:
    captured: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"message": {"content": "looks bullish"}})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        settings = LLMSettings(
            provider="ollama",
            model="gemma3:4b",
            base_url="http://localhost:11434",
            _env_file=None,  # type: ignore[call-arg]
        )
        backend = build_backend(settings, http=client)
        assert isinstance(backend, OllamaBackend)

        req_payload = LLMRequest(system="you are an analyst", user="review AAPL")
        response = await backend.complete(req_payload)

    req = captured["request"]
    assert str(req.url) == "http://localhost:11434/api/chat"
    body = json.loads(req.content)
    assert body["model"] == "gemma3:4b"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "you are an analyst"},
        {"role": "user", "content": "review AAPL"},
    ]

    assert response.text == "looks bullish"
    assert response.provider == "ollama"
    assert response.model == "gemma3:4b"


async def test_ollama_backend_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = OllamaBackend(http=client, model="gemma3:4b")
        with pytest.raises(httpx.HTTPStatusError):
            await backend.complete(LLMRequest(system="s", user="u"))


# --- aclose / client ownership --------------------------------------------------


async def test_build_backend_injected_http_aclose_leaves_it_open() -> None:
    """A client passed in via ``http=`` is owned by the caller -- ``aclose()``
    on the backend must be a no-op for it."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    async with httpx.AsyncClient(transport=transport) as client:
        settings = LLMSettings(provider="ollama", _env_file=None)  # type: ignore[call-arg]
        backend = build_backend(settings, http=client)

        await backend.aclose()

        assert client.is_closed is False


async def test_build_backend_owned_http_aclose_closes_it() -> None:
    """With no injected ``http``, ``build_backend`` creates its own client --
    ``aclose()`` must close that owned client (the resource-leak fix)."""
    settings = LLMSettings(provider="ollama", _env_file=None)  # type: ignore[call-arg]
    backend = build_backend(settings)

    assert isinstance(backend, OllamaBackend)
    assert backend.http.is_closed is False

    await backend.aclose()

    assert backend.http.is_closed is True


async def test_noop_backend_aclose_is_a_noop() -> None:
    backend = NoOpBackend()
    await backend.aclose()  # must not raise
