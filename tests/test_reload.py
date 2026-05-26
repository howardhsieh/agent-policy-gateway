"""Tests for policy hot-reload / file-watch (R20).

Acceptance criteria from the roadmap:

1. Editing the policy file flips a rule's effect for subsequent calls
   *without* restarting the process (here: without rebuilding the
   ``Gateway`` or re-reading the file by hand).
2. An invalid edit is rejected and the previous policy stays active; the
   parse error is logged (fail-closed).
"""

from __future__ import annotations

import logging
import textwrap
import time
from pathlib import Path

import pytest

from agent_policy_gateway import (
    Gateway,
    PolicyDenied,
    ToolCall,
    Verdict,
    WatchedPolicy,
    watch_policy,
)
from agent_policy_gateway.policy import PolicyError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ALLOW_YAML = textwrap.dedent(
    """\
    version: 1
    name: hot-reload-test
    rules:
      - id: gate-send-email
        when:
          tool: send_email
        effect:
          action: allow
    """
)

_DENY_YAML = textwrap.dedent(
    """\
    version: 1
    name: hot-reload-test
    rules:
      - id: gate-send-email
        when:
          tool: send_email
        effect:
          action: deny
          reason: "flipped to deny on disk"
    """
)

_INVALID_YAML = "version: 1\nname: hot-reload-test\nrules: [: : :\n"


def _write(path: Path, text: str) -> None:
    """Write ``text`` and force a distinct (mtime_ns, size) signature.

    A real operator's edit advances mtime; in a fast test two writes can land
    in the same nanosecond, so we sleep a hair to guarantee the watcher sees a
    change even when the byte length is identical.
    """
    path.write_text(text, encoding="utf-8")
    time.sleep(0.01)


def _send_email_call() -> ToolCall:
    return ToolCall(tool_name="send_email", args={"to": "ops@example.com"})


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_eager_initial_load_exposes_policy(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        wp = watch_policy(p)
        assert isinstance(wp, WatchedPolicy)
        assert wp.policy.name == "hot-reload-test"
        assert wp.path == p

    def test_invalid_starting_file_raises_immediately(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_INVALID_YAML, encoding="utf-8")
        with pytest.raises(PolicyError):
            watch_policy(p)


# ---------------------------------------------------------------------------
# acceptance criterion 1: a valid edit flips behaviour live
# ---------------------------------------------------------------------------


class TestValidReloadFlipsEffect:
    def test_first_match_reflects_edit_without_reconstruction(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        wp = watch_policy(p)
        call = _send_email_call()

        rule = wp.first_match(call)
        assert rule is not None and rule.effect.action.value == "allow"

        _write(p, _DENY_YAML)

        rule = wp.first_match(call)
        assert rule is not None and rule.effect.action.value == "deny"

    def test_gateway_decision_flips_after_edit(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        gw = Gateway(policies=[watch_policy(p)])
        call = _send_email_call()

        assert gw.decide(call).verdict == Verdict.ALLOW

        _write(p, _DENY_YAML)

        assert gw.decide(call).verdict == Verdict.DENY

    def test_gateway_execute_enforces_reloaded_deny(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        gw = Gateway(policies=[watch_policy(p)])
        call = _send_email_call()

        result, decision = gw.execute(call, lambda: "sent")
        assert result == "sent" and decision.verdict == Verdict.ALLOW

        _write(p, _DENY_YAML)

        with pytest.raises(PolicyDenied):
            gw.execute(call, lambda: "sent")

    def test_unchanged_file_does_not_reload(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        wp = watch_policy(p)
        wp.first_match(_send_email_call())
        # No write between calls => maybe_reload reports "nothing changed".
        assert wp.maybe_reload() is False

    def test_maybe_reload_returns_true_on_successful_swap(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        wp = watch_policy(p)
        _write(p, _DENY_YAML)
        assert wp.maybe_reload() is True


# ---------------------------------------------------------------------------
# acceptance criterion 2: an invalid edit is rejected, fail-closed
# ---------------------------------------------------------------------------


class TestInvalidReloadKeepsPreviousPolicy:
    def test_invalid_edit_keeps_old_policy_active(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        wp = watch_policy(p)
        call = _send_email_call()
        assert wp.first_match(call).effect.action.value == "allow"

        _write(p, _INVALID_YAML)

        # Still the previous (allow) policy, not an empty / broken one.
        rule = wp.first_match(call)
        assert rule is not None and rule.effect.action.value == "allow"

    def test_invalid_edit_then_valid_fix_recovers(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        wp = watch_policy(p)
        call = _send_email_call()

        _write(p, _INVALID_YAML)
        assert wp.first_match(call).effect.action.value == "allow"  # held

        _write(p, _DENY_YAML)
        assert wp.first_match(call).effect.action.value == "deny"  # recovered

    def test_invalid_edit_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        wp = watch_policy(p)

        _write(p, _INVALID_YAML)
        with caplog.at_level(logging.WARNING, logger="agent_policy_gateway.reload"):
            assert wp.maybe_reload() is False
        assert any(
            "policy reload failed" in r.getMessage() for r in caplog.records
        )

    def test_invalid_edit_invokes_on_error_callback(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        seen: list[Exception] = []
        wp = watch_policy(p, on_error=seen.append)

        _write(p, _INVALID_YAML)
        wp.maybe_reload()
        assert len(seen) == 1
        assert isinstance(seen[0], PolicyError)

    def test_known_bad_file_not_reparsed_every_call(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text(_ALLOW_YAML, encoding="utf-8")
        calls: list[Exception] = []
        wp = watch_policy(p, on_error=calls.append)

        _write(p, _INVALID_YAML)
        # Several calls against the same broken file => error surfaced once.
        for _ in range(5):
            wp.maybe_reload()
        assert len(calls) == 1
