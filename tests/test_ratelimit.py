"""Unit tests for the sliding-window RateLimiter (R16).

Every test drives the limiter with an injectable fake clock so window
expiry is deterministic and independent of wall-clock time.
"""

from __future__ import annotations

import pytest

from agent_policy_gateway.ratelimit import DEFAULT_WINDOW_SECONDS, RateLimiter


class FakeClock:
    """A manually advanced monotonic-style clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestConstruction:
    def test_default_window_is_sixty_seconds(self) -> None:
        assert DEFAULT_WINDOW_SECONDS == 60.0
        assert RateLimiter().window_seconds == 60.0

    def test_non_positive_window_rejected(self) -> None:
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimiter(window_seconds=0)
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimiter(window_seconds=-5)


class TestTryAcquire:
    def test_allows_up_to_limit_then_denies(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(clock=clock)
        key = ("agent.x", "web_search")
        assert rl.try_acquire(key, 3) is True
        assert rl.try_acquire(key, 3) is True
        assert rl.try_acquire(key, 3) is True
        # 4th within the window is refused
        assert rl.try_acquire(key, 3) is False

    def test_window_expiry_frees_a_slot(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(window_seconds=60, clock=clock)
        key = ("agent.x", "web_search")
        for _ in range(3):
            assert rl.try_acquire(key, 3) is True
        assert rl.try_acquire(key, 3) is False
        # Advance just shy of the window: still full.
        clock.advance(59.9)
        assert rl.try_acquire(key, 3) is False
        # Advance past the window relative to the first event: oldest expires.
        clock.advance(0.2)  # total 60.1s since the first acquire
        assert rl.try_acquire(key, 3) is True

    def test_keys_are_independent(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(clock=clock)
        a = ("agent.a", "web_search")
        b = ("agent.b", "web_search")
        assert rl.try_acquire(a, 1) is True
        assert rl.try_acquire(a, 1) is False
        # A different key has its own budget.
        assert rl.try_acquire(b, 1) is True

    def test_refused_call_does_not_consume(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(window_seconds=60, clock=clock)
        key = ("agent.x", "tool")
        assert rl.try_acquire(key, 1) is True
        # Several refusals while full...
        for _ in range(5):
            assert rl.try_acquire(key, 1) is False
        # ...then the single slot frees exactly 60s after the one accepted call.
        clock.advance(60.1)
        assert rl.try_acquire(key, 1) is True

    def test_zero_or_negative_limit_always_denies(self) -> None:
        rl = RateLimiter()
        assert rl.try_acquire(("k",), 0) is False
        assert rl.try_acquire(("k",), -1) is False


class TestPeek:
    def test_peek_is_read_only(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(clock=clock)
        key = ("agent.x", "web_search")
        # Peeking many times never consumes a slot.
        for _ in range(10):
            assert rl.peek(key, 1) is True
        # The single slot is still available to acquire.
        assert rl.try_acquire(key, 1) is True
        # Now full: peek reports False without changing state.
        assert rl.peek(key, 1) is False
        assert rl.peek(key, 1) is False

    def test_peek_reflects_consumed_slots(self) -> None:
        clock = FakeClock()
        rl = RateLimiter(clock=clock)
        key = ("agent.x", "tool")
        assert rl.peek(key, 2) is True
        rl.try_acquire(key, 2)
        rl.try_acquire(key, 2)
        assert rl.peek(key, 2) is False

    def test_peek_zero_limit_denies(self) -> None:
        assert RateLimiter().peek(("k",), 0) is False


class TestReset:
    def test_reset_single_key(self) -> None:
        rl = RateLimiter()
        a = ("a",)
        b = ("b",)
        rl.try_acquire(a, 1)
        rl.try_acquire(b, 1)
        rl.reset(a)
        assert rl.try_acquire(a, 1) is True  # freed
        assert rl.try_acquire(b, 1) is False  # untouched

    def test_reset_all(self) -> None:
        rl = RateLimiter()
        rl.try_acquire(("a",), 1)
        rl.try_acquire(("b",), 1)
        rl.reset()
        assert rl.try_acquire(("a",), 1) is True
        assert rl.try_acquire(("b",), 1) is True
