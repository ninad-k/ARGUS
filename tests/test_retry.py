"""Unit tests for ``argus.utils.retry.retry_async`` -- the generic
retry-with-backoff/timeout helper used by ``YFinanceProvider`` (kept
network-free by testing the helper directly rather than through yfinance).
"""

from __future__ import annotations

import asyncio

import pytest

from argus.utils.retry import retry_async


async def test_retry_async_succeeds_after_two_failures() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError(f"boom {calls}")
        return "ok"

    result = await retry_async(flaky, attempts=3, timeout_seconds=1.0, base_delay=0.0)

    assert result == "ok"
    assert calls == 3


async def test_retry_async_exhausts_attempts_and_raises_last_exception() -> None:
    calls = 0

    async def always_fails() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"boom {calls}")

    with pytest.raises(RuntimeError, match="boom 2"):
        await retry_async(always_fails, attempts=2, timeout_seconds=1.0, base_delay=0.0)

    assert calls == 2


async def test_retry_async_respects_timeout() -> None:
    async def hangs_forever() -> str:
        await asyncio.Event().wait()
        return "unreachable"  # pragma: no cover

    with pytest.raises(TimeoutError):
        await retry_async(hangs_forever, attempts=2, timeout_seconds=0.01, base_delay=0.0)


async def test_retry_async_retry_if_treats_empty_result_as_a_failed_attempt() -> None:
    calls = 0

    async def empty_then_full() -> list[int]:
        nonlocal calls
        calls += 1
        return [] if calls < 3 else [1, 2, 3]

    result = await retry_async(
        empty_then_full,
        attempts=3,
        timeout_seconds=1.0,
        base_delay=0.0,
        retry_if=lambda rows: len(rows) == 0,
    )

    assert result == [1, 2, 3]
    assert calls == 3


async def test_retry_async_retry_if_returns_last_empty_result_without_raising() -> None:
    """Exhausting retries on an "empty but not erroring" result must hand
    back that (possibly empty) result rather than raise -- matches the
    "provider methods never raise" contract in ``YFinanceProvider``."""
    calls = 0

    async def always_empty() -> list[int]:
        nonlocal calls
        calls += 1
        return []

    result = await retry_async(
        always_empty,
        attempts=3,
        timeout_seconds=1.0,
        base_delay=0.0,
        retry_if=lambda rows: len(rows) == 0,
    )

    assert result == []
    assert calls == 3


async def test_retry_async_on_error_hook_called_per_failed_attempt() -> None:
    seen: list[tuple[int, str]] = []

    async def always_fails() -> str:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await retry_async(
            always_fails,
            attempts=3,
            timeout_seconds=1.0,
            base_delay=0.0,
            on_error=lambda attempt, exc: seen.append((attempt, str(exc))),
        )

    assert seen == [(0, "nope"), (1, "nope"), (2, "nope")]


async def test_retry_async_first_attempt_success_short_circuits() -> None:
    calls = 0

    async def works() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await retry_async(works, attempts=5, timeout_seconds=1.0, base_delay=0.0)

    assert result == "ok"
    assert calls == 1
