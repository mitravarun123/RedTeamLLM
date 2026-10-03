"""Retry with exponential backoff and jitter."""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0, jitter: float = 0.25) -> float:
    delay = min(cap, base * (2 ** (attempt - 1)))
    return delay * (1 + random.uniform(-jitter, jitter))


def retry_call(
    fn: Callable[[], T],
    *,
    retry_on: Tuple[Type[BaseException], ...],
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.25,
    retry_after: Optional[Callable[[BaseException], Optional[float]]] = None,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except retry_on as exc:
            if attempt >= max_attempts:
                raise
            delay = backoff_delay(attempt, base_delay, max_delay, jitter)
            if retry_after is not None:
                hint = retry_after(exc)
                if hint:
                    delay = max(delay, min(float(hint), 2 * max_delay))
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            sleep(delay)