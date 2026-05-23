"""Sliding-window rate limiter for the gateway (R16).

The policy DSL allows ``action: rate_limit`` with a ``limit_per_minute``
field. This module supplies the runtime counter that turns that
declaration into enforcement: a :class:`RateLimiter` tracks the
timestamps of recent *allowed* calls, keyed by an arbitrary hashable
key (the gateway uses ``(agent_id, tool_name)``), and reports whether a
new call would exceed the configured per-window limit.

Two operations make up the contract:

* :meth:`RateLimiter.peek` — a read-only capacity check. It answers
  "would a call be allowed right now?" without recording anything, so
  the gateway's pure :meth:`~agent_policy_gateway.gateway.Gateway.decide`
  can consult it without side effects.
* :meth:`RateLimiter.try_acquire` — a check-and-consume. If the window
  has room it records the current timestamp and returns ``True``;
  otherwise it records nothing and returns ``False``. The stateful
  execution paths use this so that only calls that actually run count
  against the limit (a refused call does not consume a slot).

The clock is injectable. The default is :func:`time.monotonic`, which
is immune to wall-clock jumps (NTP steps, DST) — the right basis for a
"calls per minute" budget. Tests pass a fake clock so window expiry is
fully deterministic.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable

DEFAULT_WINDOW_SECONDS: float = 60.0

__all__ = ["DEFAULT_WINDOW_SECONDS", "RateLimiter"]


class RateLimiter:
    """A sliding-window event log, one window per key.

    A call is allowed iff the number of recorded timestamps within the
    trailing ``window_seconds`` is strictly less than the limit supplied
    at check time. Timestamps older than the window are pruned lazily on
    :meth:`try_acquire`.

    The limit is supplied per call (not at construction) because each
    policy rule carries its own ``limit_per_minute``; a single limiter
    can therefore serve rules with different limits.
    """

    def __init__(
        self,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = float(window_seconds)
        self._clock = clock
        self._events: dict[Hashable, list[float]] = {}

    @property
    def window_seconds(self) -> float:
        """The trailing window width, in seconds."""
        return self._window

    def _live_count(self, key: Hashable, now: float) -> int:
        """Count timestamps for ``key`` still inside the window at ``now``.

        Read-only: does not mutate the stored event list.
        """
        cutoff = now - self._window
        events = self._events.get(key)
        if not events:
            return 0
        return sum(1 for t in events if t > cutoff)

    def peek(self, key: Hashable, limit: int) -> bool:
        """Return True iff a call for ``key`` would be allowed right now.

        Read-only — records nothing. Used by the gateway's pure
        ``decide`` path so a dry-run verdict never consumes a slot.
        """
        if limit <= 0:
            return False
        return self._live_count(key, self._clock()) < limit

    def try_acquire(self, key: Hashable, limit: int) -> bool:
        """Consume one slot for ``key`` if the window has room.

        Returns ``True`` and records the current timestamp when the live
        count is below ``limit``; returns ``False`` and records nothing
        when the window is full. Expired timestamps for ``key`` are
        pruned as a side effect.
        """
        if limit <= 0:
            return False
        now = self._clock()
        cutoff = now - self._window
        events = self._events.get(key)
        live = [t for t in events if t > cutoff] if events else []
        if len(live) >= limit:
            # Keep the pruned list so memory does not grow unbounded even
            # when every call is refused.
            self._events[key] = live
            return False
        live.append(now)
        self._events[key] = live
        return True

    def reset(self, key: Hashable | None = None) -> None:
        """Forget recorded events.

        With no argument, clears every key. With a ``key``, clears only
        that key's window. Primarily useful in tests and for operators
        manually clearing a throttle.
        """
        if key is None:
            self._events.clear()
        else:
            self._events.pop(key, None)
