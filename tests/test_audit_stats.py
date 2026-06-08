"""Tests for ``apg audit stats`` and ``summarize_audit`` (R29).

These assert the CLI contract (exit codes 0/2/3, mirroring ``apg-replay``) and
the *stable* plain-text summary layout. The CLI is driven through ``main(argv)``
so no subprocess is needed, matching ``test_cli.py``.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from agent_policy_gateway import summarize_audit
from agent_policy_gateway.audit import (
    AuditRecord,
    JsonlAuditWriter,
    read_audit,
)
from agent_policy_gateway.cli import main
from agent_policy_gateway.core import Decision, ToolCall, Verdict


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run ``main(argv)`` capturing (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def _clock() -> Callable[[], str]:
    """A deterministic, strictly-increasing UTC-ISO clock."""
    counter: Iterator[int] = iter(range(10_000))

    def tick() -> str:
        n = next(counter)
        return f"2026-06-08T00:{n // 60:02d}:{n % 60:02d}.000000Z"

    return tick


def _write_log(
    path: Path, rows: list[tuple[str, Verdict, str | None]]
) -> Path:
    """Write ``(tool, verdict, rule_id)`` rows to a JSONL audit log."""
    writer = JsonlAuditWriter(path, clock=_clock())
    with writer:
        for tool, verdict, rule in rows:
            writer(
                ToolCall(tool_name=tool, agent_id="agent.x"),
                Decision(verdict=verdict, rule_id=rule),
            )
    return path


# --- summarize_audit (pure) ---------------------------------------------------


class TestSummarizeAudit:
    def test_empty_log_is_explained(self) -> None:
        lines = summarize_audit([], source="x.jsonl")
        assert lines[0] == "audit log summary: x.jsonl"
        assert lines[1] == "records:     0"
        assert lines[-1] == "(log is empty - no records to summarize)"

    def test_header_omits_source_when_absent(self) -> None:
        lines = summarize_audit([])
        assert lines[0] == "audit log summary"

    def _records(self) -> list[AuditRecord]:
        rows = [
            ("send_email", Verdict.DENY, "deny-web-to-email"),
            ("send_email", Verdict.DENY, "deny-web-to-email"),
            ("web_fetch", Verdict.ALLOW, None),
            ("kb_lookup", Verdict.ALLOW, "allow-internal-readers"),
            ("send_email", Verdict.REVIEW, "review-pii-egress"),
        ]
        clock = _clock()
        return [
            AuditRecord(
                ts=clock(),
                call=ToolCall(tool_name=tool, agent_id="a"),
                decision=Decision(verdict=verdict, rule_id=rule),
            )
            for tool, verdict, rule in rows
        ]

    def test_layout_is_stable(self) -> None:
        lines = summarize_audit(self._records(), source="audit.jsonl")
        assert lines[0] == "audit log summary: audit.jsonl"
        assert lines[1] == "records:     5"
        assert lines[2].startswith("span:        2026-06-08T00:00:00")
        assert "  allow  " in lines[4] and "(40.0%)" in lines[4]
        assert "  deny   " in lines[5] and "(40.0%)" in lines[5]
        assert "  review " in lines[6] and "(20.0%)" in lines[6]
        # deny+review share = (2 + 1) / 5
        assert "deny+review: 3/5  (60.0%)" in lines

    def test_top_rules_and_tools_ranked_by_hits(self) -> None:
        lines = summarize_audit(self._records())
        text = "\n".join(lines)
        assert "      2  deny-web-to-email" in text
        # unmatched decision is bucketed under the default-rule label
        assert "(default - no rule)" in text
        assert "      3  send_email" in text

    def test_top_n_truncates(self) -> None:
        lines = summarize_audit(self._records(), top_n=1)
        text = "\n".join(lines)
        # only the single most-frequent rule/tool survive
        assert "deny-web-to-email" in text
        assert "review-pii-egress" not in text
        assert "      3  send_email" in text
        assert "kb_lookup" not in text

    def test_ties_break_by_name_ascending(self) -> None:
        clock = _clock()
        recs = [
            AuditRecord(
                ts=clock(),
                call=ToolCall(tool_name=tool, agent_id="a"),
                decision=Decision(verdict=Verdict.ALLOW, rule_id=None),
            )
            for tool in ("zebra", "apple")
        ]
        lines = summarize_audit(recs)
        text = "\n".join(lines)
        assert text.index("apple") < text.index("zebra")


# --- apg audit stats (CLI contract) -------------------------------------------


class TestAuditStatsCli:
    def test_missing_file_exits_2(self, tmp_path: Path) -> None:
        rc, out, err = _run(["audit", "stats", str(tmp_path / "nope.jsonl")])
        assert rc == 2
        assert "not found" in err

    def test_malformed_log_exits_3(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json}\n", encoding="utf-8")
        rc, out, err = _run(["audit", "stats", str(bad)])
        assert rc == 3
        assert "line 1" in err

    def test_empty_log_exits_0(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        rc, out, err = _run(["audit", "stats", str(empty)])
        assert rc == 0
        assert "records:     0" in out

    def test_summary_exits_0_with_expected_lines(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "a.jsonl",
            [
                ("send_email", Verdict.DENY, "deny-web-to-email"),
                ("web_fetch", Verdict.ALLOW, None),
            ],
        )
        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert "records:     2" in out
        assert "deny+review: 1/2  (50.0%)" in out
        assert "deny-web-to-email" in out
        assert "send_email" in out

    def test_top_flag_limits_output(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "b.jsonl",
            [
                ("t_a", Verdict.ALLOW, "r1"),
                ("t_b", Verdict.ALLOW, "r2"),
                ("t_c", Verdict.ALLOW, "r3"),
            ],
        )
        rc, out, err = _run(["audit", "stats", str(log), "--top", "1"])
        assert rc == 0
        # ties break by name ascending, so only the first rule/tool survive
        assert "r1" in out and "r2" not in out and "r3" not in out
        assert "t_a" in out and "t_b" not in out and "t_c" not in out


# --- summary over the indirect-injection example log --------------------------


class TestIndirectInjectionExampleLog:
    def test_summarizes_the_gated_demo_log(self, tmp_path: Path) -> None:
        from examples.indirect_injection.gated import run_gated

        result = run_gated()
        assert result.audit, "gated demo should produce at least one record"

        log = tmp_path / "demo.jsonl"
        writer = JsonlAuditWriter(log, clock=_clock())
        with writer:
            for call, decision in result.audit:
                writer(call, decision)

        records = list(read_audit(log))
        assert len(records) == len(result.audit)

        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        # the demo's defining outcome: send_email denied by deny-web-to-email
        assert "deny-web-to-email" in out
        assert "send_email" in out
        assert f"records:     {len(result.audit)}" in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
