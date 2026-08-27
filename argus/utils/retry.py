"""Small async retry-with-backoff helper shared by data providers.

Generic so any provider (not just yfinance) can bound a flaky/slow call with
a per-attempt timeout, exponential backoff, and a little jitter, without
every provider re-implementing the loop.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    timeout_seconds: float,
    base_delay: float = 1.0,
    retry_if: Callable[[T], bool] | None = None,
    on_error: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call ``fn()`` up to ``attempts`` times, each bounded by ``timeout_seconds``.

    Each attempt runs under ``asyncio.wait_for(fn(), timeout=timeout_seconds)``
    -- a slow ``fn()`` becomes a ``TimeoutError`` rather than hanging the
    caller forever. Note that ``wait_for`` only abandons the *awaitable*; if
    ``fn()`` wraps ``asyncio.to_thread(...)``, a genuinely stuck OS thread
    keeps running in the background after this returns. That's an accepted,
    documented limitation -- it still stops the timeout from propagating up
    and hanging the caller, which is the actual goal.

    Any exception (including that ``TimeoutError``) triggers a retry. If
    ``retry_if`` is given, a *successful* result also triggers a retry when
    ``retry_if(result)`` is true (e.g. an empty bars array from a flaky
    upstream) -- except on the last attempt, whose result is always returned
    as-is, empty or not.

    Retries back off ``base_delay * 2**attempt`` seconds plus a little
    jitter. ``on_error(attempt, exc)`` is invoked for each attempt that
    raised (not for a ``retry_if``-triggered empty result), letting callers
    log with their own context.

    Raises the last exception if every attempt raised. Never raises purely
    because of ``retry_if`` -- the last (possibly "empty") result is
    returned instead.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None
    for attempt in range(attempts):
        is_last = attempt == attempts - 1
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001 -- any failure is retryable here
            last_exc = exc
            if on_error is not None:
                on_error(attempt, exc)
        else:
            if retry_if is None or not retry_if(result) or is_last:
                return result

        if not is_last:
            jitter = random.uniform(0, base_delay * 0.1)  # noqa: S311 -- backoff jitter, not crypto
            await asyncio.sleep(base_delay * (2**attempt) + jitter)

    assert last_exc is not None  # every path that falls out of the loop above set it
    raise last_exc
