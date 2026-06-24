"""Tests for ``apg audit stats`` and ``summarize_audit`` (R29).

These assert the CLI contract (exit codes 0/2/3, mirroring ``apg-replay``) and
the *stable* plain-text summary layout. The CLI is driven through ``main(argv)``
so no subprocess is needed, matching ``test_cli.py``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from agent_policy_gateway import (
    audit_stats_csv,
    audit_stats_dict,
    filter_by_agent,
    filter_by_rule,
    filter_by_time,
    filter_by_tool,
    filter_by_verdict,
    summarize_audit,
)
from agent_policy_gateway.audit import (
    _NO_AGENT,
    _NO_RULE,
    AuditRecord,
    JsonlAuditWriter,
    audit_flagged_share,
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


# --- audit_stats_dict (pure) + --json CLI path (R30) --------------------------


def _mixed_records() -> list[AuditRecord]:
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


class TestAuditStatsDict:
    def test_empty_log_yields_source_and_zero_records_only(self) -> None:
        d = audit_stats_dict([], source="x.jsonl")
        assert d == {"source": "x.jsonl", "records": 0}

    def test_source_omitted_when_absent(self) -> None:
        d = audit_stats_dict([])
        assert d == {"records": 0}
        assert "source" not in d

    def test_counts_and_percentages(self) -> None:
        d = audit_stats_dict(_mixed_records(), source="log.jsonl", top_n=5)
        assert d["source"] == "log.jsonl"
        assert d["records"] == 5
        # all four verdicts present, in enum order, even at zero hits
        assert list(d["verdicts"]) == ["allow", "deny", "review", "redact"]
        assert d["verdicts"]["allow"] == {"count": 2, "pct": 40.0}
        assert d["verdicts"]["deny"] == {"count": 2, "pct": 40.0}
        assert d["verdicts"]["review"] == {"count": 1, "pct": 20.0}
        assert d["verdicts"]["redact"] == {"count": 0, "pct": 0.0}
        assert d["deny_review"] == {"count": 3, "pct": 60.0}

    def test_span_is_min_and_max_timestamp(self) -> None:
        recs = _mixed_records()
        d = audit_stats_dict(recs)
        assert d["span"]["first"] == min(r.ts for r in recs)
        assert d["span"]["last"] == max(r.ts for r in recs)

    def test_top_rules_and_tools_ordered_by_hits(self) -> None:
        d = audit_stats_dict(_mixed_records())
        assert d["top_rules"][0] == {"name": "deny-web-to-email", "count": 2}
        assert d["top_tools"][0] == {"name": "send_email", "count": 3}
        # the no-rule decision is bucketed under the default label
        rule_names = {r["name"] for r in d["top_rules"]}
        assert "(default - no rule)" in rule_names

    def test_top_n_limits_lists(self) -> None:
        d = audit_stats_dict(_mixed_records(), top_n=1)
        assert len(d["top_rules"]) == 1
        assert len(d["top_tools"]) == 1

    def test_is_json_serializable(self) -> None:
        d = audit_stats_dict(_mixed_records(), source="log.jsonl")
        # round-trips through JSON unchanged
        assert json.loads(json.dumps(d)) == d


class TestAuditStatsJsonCli:
    def test_json_flag_emits_valid_json(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "a.jsonl",
            [
                ("send_email", Verdict.DENY, "deny-web-to-email"),
                ("web_fetch", Verdict.ALLOW, None),
            ],
        )
        rc, out, err = _run(["audit", "stats", str(log), "--json"])
        assert rc == 0
        assert err == ""
        payload = json.loads(out)
        assert payload["records"] == 2
        assert payload["source"] == str(log)
        assert payload["verdicts"]["deny"]["count"] == 1
        assert payload["verdicts"]["allow"]["count"] == 1

    def test_json_respects_top_flag(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "b.jsonl",
            [
                ("t1", Verdict.ALLOW, "r1"),
                ("t2", Verdict.ALLOW, "r2"),
                ("t3", Verdict.ALLOW, "r3"),
            ],
        )
        rc, out, _ = _run(["audit", "stats", str(log), "--json", "--top", "2"])
        assert rc == 0
        payload = json.loads(out)
        assert len(payload["top_tools"]) == 2
        assert len(payload["top_rules"]) == 2

    def test_json_empty_log(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.jsonl"
        log.write_text("", encoding="utf-8")
        rc, out, _ = _run(["audit", "stats", str(log), "--json"])
        assert rc == 0
        assert json.loads(out) == {"source": str(log), "records": 0}

    def test_json_missing_file_exits_2(self, tmp_path: Path) -> None:
        rc, out, err = _run(
            ["audit", "stats", str(tmp_path / "nope.jsonl"), "--json"]
        )
        assert rc == 2
        assert out == ""
        assert "not found" in err

    def test_json_malformed_log_exits_3(self, tmp_path: Path) -> None:
        log = tmp_path / "bad.jsonl"
        log.write_text("{not json}\n", encoding="utf-8")
        rc, out, err = _run(["audit", "stats", str(log), "--json"])
        assert rc == 3
        assert out == ""


# --- top agents breakdown (R33) -----------------------------------------------


def _agent_records(
    agents: list[str | None],
) -> list[AuditRecord]:
    """One ALLOW record per entry, varying only ``agent_id``."""
    clock = _clock()
    return [
        AuditRecord(
            ts=clock(),
            call=ToolCall(tool_name="t", agent_id=agent),
            decision=Decision(verdict=Verdict.ALLOW, rule_id=None),
        )
        for agent in agents
    ]


class TestTopAgentsSummary:
    def test_text_block_ranks_agents_by_hits(self) -> None:
        recs = _agent_records(["beta", "beta", "alpha", "beta", "alpha"])
        lines = summarize_audit(recs)
        text = "\n".join(lines)
        assert "top agents (by hits, max 5):" in text
        assert "      3  beta" in text
        assert "      2  alpha" in text
        # busiest agent listed before the less-busy one
        assert text.index("  beta") < text.index("  alpha")

    def test_missing_agent_id_bucketed_under_label(self) -> None:
        recs = _agent_records(["alpha", None, None])
        text = "\n".join(summarize_audit(recs))
        assert "      2  (unattributed - no agent_id)" in text
        assert "      1  alpha" in text

    def test_ties_break_by_name_ascending(self) -> None:
        recs = _agent_records(["zebra", "apple"])
        text = "\n".join(summarize_audit(recs))
        block = text.split("top agents")[1]
        assert block.index("apple") < block.index("zebra")

    def test_top_n_truncates_agent_block(self) -> None:
        recs = _agent_records(["a", "a", "b", "c"])
        text = "\n".join(summarize_audit(recs, top_n=1))
        block = text.split("top agents")[1]
        assert "  a" in block
        assert "  b" not in block and "  c" not in block

    def test_empty_log_has_no_agent_block(self) -> None:
        lines = summarize_audit([], source="x.jsonl")
        assert all("top agents" not in line for line in lines)


class TestTopAgentsDict:
    def test_top_agents_ordered_by_hits(self) -> None:
        d = audit_stats_dict(_agent_records(["beta", "beta", "alpha"]))
        assert d["top_agents"][0] == {"name": "beta", "count": 2}
        assert d["top_agents"][1] == {"name": "alpha", "count": 1}

    def test_missing_agent_id_bucketed(self) -> None:
        d = audit_stats_dict(_agent_records(["alpha", None, None]))
        names = {a["name"]: a["count"] for a in d["top_agents"]}
        assert names["(unattributed - no agent_id)"] == 2

    def test_top_n_limits_agent_list(self) -> None:
        d = audit_stats_dict(_agent_records(["a", "b", "c"]), top_n=1)
        assert len(d["top_agents"]) == 1

    def test_empty_log_has_no_top_agents_key(self) -> None:
        d = audit_stats_dict([], source="x.jsonl")
        assert "top_agents" not in d

    def test_is_json_serializable(self) -> None:
        d = audit_stats_dict(_agent_records(["a", "b", None]))
        assert json.loads(json.dumps(d)) == d


class TestTopAgentsCli:
    def test_text_cli_shows_agent_block(self, tmp_path: Path) -> None:
        log = tmp_path / "agents.jsonl"
        writer = JsonlAuditWriter(log, clock=_clock())
        with writer:
            for agent in ("svc.alpha", "svc.alpha", "svc.beta"):
                writer(
                    ToolCall(tool_name="send_email", agent_id=agent),
                    Decision(verdict=Verdict.ALLOW, rule_id=None),
                )
        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert "top agents (by hits, max 5):" in out
        assert "      2  svc.alpha" in out
        assert "      1  svc.beta" in out

    def test_json_cli_includes_top_agents(self, tmp_path: Path) -> None:
        log = tmp_path / "agents.jsonl"
        writer = JsonlAuditWriter(log, clock=_clock())
        with writer:
            for agent in ("svc.alpha", "svc.alpha", "svc.beta"):
                writer(
                    ToolCall(tool_name="send_email", agent_id=agent),
                    Decision(verdict=Verdict.ALLOW, rule_id=None),
                )
        rc, out, _ = _run(["audit", "stats", str(log), "--json"])
        assert rc == 0
        payload = json.loads(out)
        assert payload["top_agents"][0] == {"name": "svc.alpha", "count": 2}


# --- verdict filter (R31) -----------------------------------------------------


def _mixed_log(path: Path) -> Path:
    """A log with every verdict represented at least once."""
    return _write_log(
        path,
        [
            ("send_email", Verdict.DENY, "deny-web-to-email"),
            ("send_email", Verdict.DENY, "deny-web-to-email"),
            ("web_fetch", Verdict.ALLOW, None),
            ("kb_lookup", Verdict.ALLOW, "allow-internal-readers"),
            ("send_email", Verdict.REVIEW, "review-pii-egress"),
            ("scrub_tool", Verdict.REDACT, "redact-pii"),
        ],
    )


class TestFilterByVerdict:
    def _records(self) -> list[AuditRecord]:
        clock = _clock()
        rows = [
            ("send_email", Verdict.DENY, "deny"),
            ("web_fetch", Verdict.ALLOW, None),
            ("send_email", Verdict.REVIEW, "review"),
            ("scrub", Verdict.REDACT, "redact"),
        ]
        return [
            AuditRecord(
                ts=clock(),
                call=ToolCall(tool_name=t, agent_id="a"),
                decision=Decision(verdict=v, rule_id=r),
            )
            for t, v, r in rows
        ]

    def test_none_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_verdict(recs, None) == recs

    def test_empty_iterable_returns_all(self) -> None:
        recs = self._records()
        assert filter_by_verdict(recs, []) == recs

    def test_single_verdict_keeps_only_matches(self) -> None:
        recs = self._records()
        out = filter_by_verdict(recs, ["deny"])
        assert [r.decision.verdict for r in out] == [Verdict.DENY]

    def test_accepts_enum_members(self) -> None:
        recs = self._records()
        out = filter_by_verdict(recs, [Verdict.ALLOW])
        assert [r.decision.verdict for r in out] == [Verdict.ALLOW]

    def test_multiple_verdicts_union(self) -> None:
        recs = self._records()
        out = filter_by_verdict(recs, ["deny", "review"])
        assert {r.decision.verdict for r in out} == {Verdict.DENY, Verdict.REVIEW}

    def test_no_match_yields_empty_list(self) -> None:
        # a log with no redact records filtered to redact -> empty
        recs = [r for r in self._records() if r.decision.verdict is not Verdict.REDACT]
        assert filter_by_verdict(recs, ["redact"]) == []

    def test_preserves_order(self) -> None:
        recs = self._records()
        out = filter_by_verdict(recs, ["deny", "allow", "review", "redact"])
        assert out == recs


class TestAuditStatsVerdictCli:
    def test_text_filter_to_deny(self, tmp_path: Path) -> None:
        log = _mixed_log(tmp_path / "m.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--verdict", "deny"])
        assert rc == 0
        # only the two deny records are summarized
        assert "records:     2" in out
        assert "deny+review: 2/2  (100.0%)" in out
        # allow-only tools/rules from the full log do not appear
        assert "web_fetch" not in out
        assert "allow-internal-readers" not in out

    def test_json_filter_to_deny(self, tmp_path: Path) -> None:
        log = _mixed_log(tmp_path / "m.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--json", "--verdict", "deny"])
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 2
        assert data["verdicts"]["deny"]["count"] == 2
        assert data["verdicts"]["allow"]["count"] == 0
        names = {t["name"] for t in data["top_tools"]}
        assert names == {"send_email"}

    def test_repeatable_verdict_union(self, tmp_path: Path) -> None:
        log = _mixed_log(tmp_path / "m.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--json", "--verdict", "deny", "--verdict", "review"]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 3  # 2 deny + 1 review
        assert data["verdicts"]["deny"]["count"] == 2
        assert data["verdicts"]["review"]["count"] == 1

    def test_empty_after_filter_summarizes_as_empty_log(self, tmp_path: Path) -> None:
        # a log with no review records, filtered to review
        log = _write_log(
            tmp_path / "noreview.jsonl",
            [
                ("send_email", Verdict.DENY, "deny"),
                ("web_fetch", Verdict.ALLOW, None),
            ],
        )
        rc, out, err = _run(["audit", "stats", str(log), "--verdict", "review"])
        assert rc == 0
        assert "records:     0" in out
        assert "(log is empty - no records to summarize)" in out

    def test_empty_after_filter_json(self, tmp_path: Path) -> None:
        log = _write_log(
            tmp_path / "noreview.jsonl",
            [("send_email", Verdict.DENY, "deny")],
        )
        rc, out, err = _run(["audit", "stats", str(log), "--json", "--verdict", "review"])
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 0
        assert "verdicts" not in data  # empty-log shortcut

    def test_unknown_verdict_is_choices_error_exit_2(self, tmp_path: Path) -> None:
        log = _mixed_log(tmp_path / "m.jsonl")
        with pytest.raises(SystemExit) as exc:
            _run(["audit", "stats", str(log), "--verdict", "bogus"])
        assert exc.value.code == 2


# --- R32: reading the log from stdin (``apg audit stats -``) ------------------


class TestAuditStatsStdin:
    """``-`` as the log positional reads JSONL from ``sys.stdin``."""

    def _feed(self, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(text))

    def test_dash_matches_file_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _write_log(
            tmp_path / "a.jsonl",
            [
                ("send_email", Verdict.DENY, "deny-web-to-email"),
                ("web_fetch", Verdict.ALLOW, None),
            ],
        )
        rc_file, out_file, _ = _run(["audit", "stats", str(log)])
        assert rc_file == 0

        self._feed(monkeypatch, log.read_text())
        rc_pipe, out_pipe, _ = _run(["audit", "stats", "-"])
        assert rc_pipe == 0
        # Only the header source line differs; everything below is identical.
        assert out_file.splitlines()[1:] == out_pipe.splitlines()[1:]

    def test_header_shows_stdin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _write_log(
            tmp_path / "a.jsonl", [("web_fetch", Verdict.ALLOW, None)]
        )
        self._feed(monkeypatch, log.read_text())
        rc, out, err = _run(["audit", "stats", "-"])
        assert rc == 0
        assert out.splitlines()[0] == "audit log summary: <stdin>"

    def test_malformed_piped_line_exits_3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._feed(monkeypatch, "{ this is not json }\n")
        rc, out, err = _run(["audit", "stats", "-"])
        assert rc == 3
        assert "line 1" in err

    def test_json_over_stdin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _write_log(
            tmp_path / "a.jsonl",
            [
                ("send_email", Verdict.DENY, "deny-web-to-email"),
                ("web_fetch", Verdict.ALLOW, None),
            ],
        )
        self._feed(monkeypatch, log.read_text())
        rc, out, err = _run(["audit", "stats", "-", "--json"])
        assert rc == 0
        data = json.loads(out)
        assert data["source"] == "<stdin>"
        assert data["records"] == 2

    def test_empty_stdin_yields_zero_records(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._feed(monkeypatch, "")
        rc, out, err = _run(["audit", "stats", "-"])
        assert rc == 0
        assert "records:     0" in out



# --- R34: timestamp-window filter (``--since`` / ``--until``) ------------------


class TestFilterByTime:
    """Pure ``filter_by_time`` helper: inclusive, lexicographic ISO bounds."""

    def _records(self) -> list[AuditRecord]:
        tss = [
            "2026-06-08T00:00:00.000000Z",
            "2026-06-09T12:00:00.000000Z",
            "2026-06-10T23:59:59.000000Z",
            "2026-06-11T06:30:00.000000Z",
        ]
        return [
            AuditRecord(
                ts=ts,
                call=ToolCall(tool_name="t", agent_id="a"),
                decision=Decision(verdict=Verdict.ALLOW, rule_id=None),
            )
            for ts in tss
        ]

    def test_no_bounds_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_time(recs) == recs

    def test_since_is_inclusive_lower_bound(self) -> None:
        recs = self._records()
        out = filter_by_time(recs, since="2026-06-09T12:00:00.000000Z")
        assert [r.ts for r in out] == [
            "2026-06-09T12:00:00.000000Z",
            "2026-06-10T23:59:59.000000Z",
            "2026-06-11T06:30:00.000000Z",
        ]

    def test_until_is_inclusive_upper_bound(self) -> None:
        recs = self._records()
        out = filter_by_time(recs, until="2026-06-10T23:59:59.000000Z")
        assert [r.ts for r in out] == [
            "2026-06-08T00:00:00.000000Z",
            "2026-06-09T12:00:00.000000Z",
            "2026-06-10T23:59:59.000000Z",
        ]

    def test_since_and_until_window(self) -> None:
        recs = self._records()
        out = filter_by_time(
            recs,
            since="2026-06-09T00:00:00.000000Z",
            until="2026-06-10T23:59:59.000000Z",
        )
        assert [r.ts for r in out] == [
            "2026-06-09T12:00:00.000000Z",
            "2026-06-10T23:59:59.000000Z",
        ]

    def test_iso_date_prefix_works_as_bound(self) -> None:
        # a bare date prefix selects from the start of that day onward
        recs = self._records()
        out = filter_by_time(recs, since="2026-06-10")
        assert [r.ts for r in out] == [
            "2026-06-10T23:59:59.000000Z",
            "2026-06-11T06:30:00.000000Z",
        ]

    def test_window_matching_nothing_is_empty(self) -> None:
        recs = self._records()
        assert filter_by_time(recs, since="2027-01-01T00:00:00.000000Z") == []

    def test_preserves_order(self) -> None:
        recs = self._records()
        out = filter_by_time(recs, since="2026-06-08T00:00:00.000000Z")
        assert out == recs


class TestAuditStatsTimeCli:
    """``apg audit stats --since/--until`` over a real JSONL log."""

    def _log(self, path: Path) -> Path:
        # _clock() yields 2026-06-08T00:00:00, ...:01, ...:02, ...:03
        return _write_log(
            path,
            [
                ("send_email", Verdict.DENY, "deny"),
                ("web_fetch", Verdict.ALLOW, None),
                ("kb_lookup", Verdict.ALLOW, "allow"),
                ("send_email", Verdict.REVIEW, "review"),
            ],
        )

    def test_since_filters_text(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--since", "2026-06-08T00:00:02.000000Z"]
        )
        assert rc == 0
        assert "records:     2" in out

    def test_until_filters_text(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--until", "2026-06-08T00:00:01.000000Z"]
        )
        assert rc == 0
        assert "records:     2" in out

    def test_window_json_scopes_span(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--since", "2026-06-08T00:00:01.000000Z",
                "--until", "2026-06-08T00:00:02.000000Z",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 2
        assert data["span"]["first"] == "2026-06-08T00:00:01.000000Z"
        assert data["span"]["last"] == "2026-06-08T00:00:02.000000Z"

    def test_composes_with_verdict(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        # since 00:00:01 keeps web_fetch(allow)/kb_lookup(allow)/send_email(review),
        # then --verdict allow narrows to the two allow records.
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--since", "2026-06-08T00:00:01.000000Z",
                "--verdict", "allow",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 2
        assert data["verdicts"]["allow"]["count"] == 2

    def test_window_matching_nothing_is_empty_log(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--since", "2027-01-01T00:00:00.000000Z"]
        )
        assert rc == 0
        assert "records:     0" in out
        assert "(log is empty - no records to summarize)" in out

    def test_no_window_matches_all(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert "records:     4" in out



class TestFilterByTool:
    """Pure ``filter_by_tool`` helper: fnmatch globs over ``call.tool_name``."""

    def _records(self) -> list[AuditRecord]:
        names = ["send_email", "send_sms", "web_fetch", "kb_lookup"]
        return [
            AuditRecord(
                ts="2026-06-08T00:00:00.000000Z",
                call=ToolCall(tool_name=name, agent_id="a"),
                decision=Decision(verdict=Verdict.ALLOW, rule_id=None),
            )
            for name in names
        ]

    def test_no_filter_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_tool(recs, None) == recs

    def test_empty_patterns_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_tool(recs, []) == recs

    def test_single_glob_matches_prefix(self) -> None:
        recs = self._records()
        out = filter_by_tool(recs, ["send_*"])
        assert [r.call.tool_name for r in out] == ["send_email", "send_sms"]

    def test_literal_pattern_is_exact_match(self) -> None:
        recs = self._records()
        out = filter_by_tool(recs, ["web_fetch"])
        assert [r.call.tool_name for r in out] == ["web_fetch"]

    def test_multi_pattern_union(self) -> None:
        recs = self._records()
        out = filter_by_tool(recs, ["web_fetch", "kb_*"])
        assert [r.call.tool_name for r in out] == ["web_fetch", "kb_lookup"]

    def test_no_match_is_empty(self) -> None:
        recs = self._records()
        assert filter_by_tool(recs, ["nope_*"]) == []

    def test_matching_is_case_sensitive(self) -> None:
        recs = self._records()
        assert filter_by_tool(recs, ["SEND_*"]) == []

    def test_preserves_order(self) -> None:
        recs = self._records()
        out = filter_by_tool(recs, ["*"])
        assert out == recs


class TestAuditStatsToolCli:
    """``apg audit stats --tool`` over a real JSONL log."""

    def _log(self, path: Path) -> Path:
        return _write_log(
            path,
            [
                ("send_email", Verdict.DENY, "deny"),
                ("web_fetch", Verdict.ALLOW, None),
                ("kb_lookup", Verdict.ALLOW, "allow"),
                ("send_email", Verdict.REVIEW, "review"),
            ],
        )

    def test_glob_filters_text(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--tool", "send_*"])
        assert rc == 0
        assert "records:     2" in out

    def test_glob_filters_json(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--json", "--tool", "send_*"]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 2
        assert data["verdicts"]["deny"]["count"] == 1
        assert data["verdicts"]["review"]["count"] == 1

    def test_repeatable_tool_unions(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--tool", "web_fetch", "--tool", "kb_lookup",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 2
        assert data["verdicts"]["allow"]["count"] == 2

    def test_composes_with_verdict(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        # send_* matches the deny+review rows; --verdict deny narrows to one.
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--tool", "send_*", "--verdict", "deny",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 1
        assert data["verdicts"]["deny"]["count"] == 1

    def test_no_match_is_empty_log(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--tool", "nope_*"])
        assert rc == 0
        assert "records:     0" in out
        assert "(log is empty - no records to summarize)" in out

    def test_no_tool_matches_all(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert "records:     4" in out


class TestFilterByAgent:
    """Pure ``filter_by_agent`` helper: fnmatch globs over ``call.agent_id``."""

    def _records(self) -> list[AuditRecord]:
        # agent ids include the unattributed (None) bucket.
        agents: list[str | None] = ["svc.mailer", "svc.crawler", "user.alice", None]
        return [
            AuditRecord(
                ts="2026-06-08T00:00:00.000000Z",
                call=ToolCall(tool_name="t", agent_id=aid),
                decision=Decision(verdict=Verdict.ALLOW, rule_id=None),
            )
            for aid in agents
        ]

    def test_no_filter_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_agent(recs, None) == recs

    def test_empty_patterns_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_agent(recs, []) == recs

    def test_single_glob_matches_prefix(self) -> None:
        recs = self._records()
        out = filter_by_agent(recs, ["svc.*"])
        assert [r.call.agent_id for r in out] == ["svc.mailer", "svc.crawler"]

    def test_literal_pattern_is_exact_match(self) -> None:
        recs = self._records()
        out = filter_by_agent(recs, ["user.alice"])
        assert [r.call.agent_id for r in out] == ["user.alice"]

    def test_multi_pattern_union(self) -> None:
        recs = self._records()
        out = filter_by_agent(recs, ["user.alice", "svc.crawler"])
        assert [r.call.agent_id for r in out] == ["svc.crawler", "user.alice"]

    def test_unattributed_bucket_selected_by_sentinel(self) -> None:
        recs = self._records()
        out = filter_by_agent(recs, [_NO_AGENT])
        assert [r.call.agent_id for r in out] == [None]

    def test_sentinel_composes_with_named_agent(self) -> None:
        recs = self._records()
        out = filter_by_agent(recs, ["user.alice", _NO_AGENT])
        assert [r.call.agent_id for r in out] == ["user.alice", None]

    def test_no_match_is_empty(self) -> None:
        recs = self._records()
        assert filter_by_agent(recs, ["nope.*"]) == []

    def test_matching_is_case_sensitive(self) -> None:
        recs = self._records()
        assert filter_by_agent(recs, ["SVC.*"]) == []

    def test_preserves_order(self) -> None:
        recs = self._records()
        out = filter_by_agent(recs, ["*"])
        # "*" matches every named agent; the None bucket maps to the sentinel
        # label, which "*" also matches, so all records pass through unchanged.
        assert out == recs


class TestAuditStatsAgentCli:
    """``apg audit stats --agent`` over a real JSONL log."""

    def _log(self, path: Path) -> Path:
        rows: list[tuple[str, Verdict, str | None]] = [
            ("send_email", Verdict.DENY, "deny"),
            ("web_fetch", Verdict.ALLOW, None),
            ("kb_lookup", Verdict.ALLOW, "allow"),
            ("send_email", Verdict.REVIEW, "review"),
        ]
        agents: list[str | None] = ["svc.mailer", None, "user.alice", "svc.mailer"]
        writer = JsonlAuditWriter(path, clock=_clock())
        with writer:
            for (tool, verdict, rule), aid in zip(rows, agents, strict=True):
                writer(
                    ToolCall(tool_name=tool, agent_id=aid),
                    Decision(verdict=verdict, rule_id=rule),
                )
        return path

    def test_glob_filters_text(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--agent", "svc.*"])
        assert rc == 0
        assert "records:     2" in out

    def test_glob_filters_json(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--json", "--agent", "svc.mailer"]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 2
        assert data["verdicts"]["deny"]["count"] == 1
        assert data["verdicts"]["review"]["count"] == 1

    def test_repeatable_agent_unions(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--agent", "user.alice", "--agent", "svc.mailer",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 3

    def test_sentinel_selects_unattributed(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--agent", _NO_AGENT,
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 1
        assert data["verdicts"]["allow"]["count"] == 1

    def test_composes_with_tool(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        # svc.mailer has the two send_email rows; --tool send_* keeps both,
        # --verdict deny narrows to one.
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--agent", "svc.mailer", "--tool", "send_*", "--verdict", "deny",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 1
        assert data["verdicts"]["deny"]["count"] == 1

    def test_no_match_is_empty_log(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--agent", "nope.*"])
        assert rc == 0
        assert "records:     0" in out
        assert "(log is empty - no records to summarize)" in out

    def test_no_agent_matches_all(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert "records:     4" in out


# --- R37: union of several logs (``apg audit stats a.jsonl b.jsonl``) ---------


class TestAuditStatsMultiLog:
    """``stats`` accepts >1 ``log`` positional and summarizes their union."""

    def test_two_files_record_count_is_the_sum(self, tmp_path: Path) -> None:
        a = _write_log(
            tmp_path / "a.jsonl",
            [
                ("web_fetch", Verdict.ALLOW, None),
                ("send_email", Verdict.DENY, "deny-web-to-email"),
            ],
        )
        b = _write_log(
            tmp_path / "b.jsonl",
            [("send_email", Verdict.DENY, "deny-web-to-email")],
        )
        rc, out, err = _run(["audit", "stats", str(a), str(b), "--json"])
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 3
        assert data["verdicts"]["deny"]["count"] == 2
        assert data["verdicts"]["allow"]["count"] == 1

    def test_union_span_spans_both_logs(self, tmp_path: Path) -> None:
        # Each ``_write_log`` uses its own clock starting at 00:00:00, so the
        # two files share the same timestamp range; build distinct ranges by
        # hand so the union span is provably wider than either file alone.
        early = tmp_path / "early.jsonl"
        late = tmp_path / "late.jsonl"
        writer = JsonlAuditWriter(early, clock=lambda: "2026-06-01T00:00:00.000000Z")
        with writer:
            writer(
                ToolCall(tool_name="web_fetch", agent_id="a"),
                Decision(verdict=Verdict.ALLOW, rule_id=None),
            )
        writer = JsonlAuditWriter(late, clock=lambda: "2026-06-30T23:59:59.000000Z")
        with writer:
            writer(
                ToolCall(tool_name="send_email", agent_id="a"),
                Decision(verdict=Verdict.DENY, rule_id="r"),
            )
        rc, out, err = _run(["audit", "stats", str(early), str(late), "--json"])
        assert rc == 0
        span = json.loads(out)["span"]
        assert span["first"] == "2026-06-01T00:00:00.000000Z"
        assert span["last"] == "2026-06-30T23:59:59.000000Z"

    def test_records_read_in_argument_order(self, tmp_path: Path) -> None:
        # Argument order, not file name, decides concatenation order. The text
        # span uses min/max so it is order-insensitive; assert union size and
        # that swapping order yields an identical summary body.
        a = _write_log(tmp_path / "a.jsonl", [("t1", Verdict.ALLOW, None)])
        b = _write_log(tmp_path / "b.jsonl", [("t2", Verdict.DENY, "r")])
        rc1, out1, _ = _run(["audit", "stats", str(a), str(b), "--json"])
        rc2, out2, _ = _run(["audit", "stats", str(b), str(a), "--json"])
        assert rc1 == 0 and rc2 == 0
        d1, d2 = json.loads(out1), json.loads(out2)
        assert d1["records"] == d2["records"] == 2

    def test_source_label_is_comma_joined(self, tmp_path: Path) -> None:
        a = _write_log(tmp_path / "a.jsonl", [("t", Verdict.ALLOW, None)])
        b = _write_log(tmp_path / "b.jsonl", [("t", Verdict.ALLOW, None)])
        rc, out, err = _run(["audit", "stats", str(a), str(b)])
        assert rc == 0
        assert out.splitlines()[0] == f"audit log summary: {a}, {b}"

    def test_json_source_label_is_comma_joined(self, tmp_path: Path) -> None:
        a = _write_log(tmp_path / "a.jsonl", [("t", Verdict.ALLOW, None)])
        b = _write_log(tmp_path / "b.jsonl", [("t", Verdict.ALLOW, None)])
        rc, out, err = _run(["audit", "stats", str(a), str(b), "--json"])
        assert rc == 0
        assert json.loads(out)["source"] == f"{a}, {b}"

    def test_many_logs_collapse_to_n_logs_label(self, tmp_path: Path) -> None:
        logs = [
            str(_write_log(tmp_path / f"l{i}.jsonl", [("t", Verdict.ALLOW, None)]))
            for i in range(5)
        ]
        rc, out, err = _run(["audit", "stats", *logs, "--json"])
        assert rc == 0
        data = json.loads(out)
        assert data["source"] == "5 logs"
        assert data["records"] == 5

    def test_single_file_behavior_unchanged(self, tmp_path: Path) -> None:
        # One positional must summarize byte-for-byte as before R37: the source
        # label is the bare path, not a one-element joined list.
        log = _write_log(
            tmp_path / "a.jsonl",
            [("web_fetch", Verdict.ALLOW, None)],
        )
        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert out.splitlines()[0] == f"audit log summary: {log}"

    def test_filters_apply_to_the_union(self, tmp_path: Path) -> None:
        a = _write_log(
            tmp_path / "a.jsonl",
            [
                ("web_fetch", Verdict.ALLOW, None),
                ("send_email", Verdict.DENY, "r"),
            ],
        )
        b = _write_log(
            tmp_path / "b.jsonl",
            [("send_email", Verdict.DENY, "r")],
        )
        rc, out, err = _run(
            ["audit", "stats", str(a), str(b), "--verdict", "deny", "--json"]
        )
        assert rc == 0
        assert json.loads(out)["records"] == 2

    def test_missing_file_among_several_exits_2_naming_it(
        self, tmp_path: Path
    ) -> None:
        a = _write_log(tmp_path / "a.jsonl", [("t", Verdict.ALLOW, None)])
        missing = tmp_path / "gone.jsonl"
        rc, out, err = _run(["audit", "stats", str(a), str(missing)])
        assert rc == 2
        assert "gone.jsonl" in err
        assert "not found" in err

    def test_dash_mixed_with_path_exits_2(self, tmp_path: Path) -> None:
        a = _write_log(tmp_path / "a.jsonl", [("t", Verdict.ALLOW, None)])
        rc, out, err = _run(["audit", "stats", "-", str(a)])
        assert rc == 2
        assert "-" in err and "cannot be combined" in err

    def test_dash_after_path_also_exits_2(self, tmp_path: Path) -> None:
        a = _write_log(tmp_path / "a.jsonl", [("t", Verdict.ALLOW, None)])
        rc, out, err = _run(["audit", "stats", str(a), "-"])
        assert rc == 2
        assert "cannot be combined" in err

    def test_malformed_line_in_second_log_exits_3(self, tmp_path: Path) -> None:
        a = _write_log(tmp_path / "a.jsonl", [("t", Verdict.ALLOW, None)])
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{ not json }\n")
        rc, out, err = _run(["audit", "stats", str(a), str(bad)])
        assert rc == 3
        assert "line 1" in err

    def test_single_dash_still_reads_stdin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _write_log(
            tmp_path / "a.jsonl",
            [("send_email", Verdict.DENY, "r"), ("web_fetch", Verdict.ALLOW, None)],
        )
        monkeypatch.setattr("sys.stdin", io.StringIO(log.read_text()))
        rc, out, err = _run(["audit", "stats", "-", "--json"])
        assert rc == 0
        data = json.loads(out)
        assert data["source"] == "<stdin>"
        assert data["records"] == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))


# --- R38: per-list top-N overrides -------------------------------------------


def _distinct_records(n: int) -> list[AuditRecord]:
    """``n`` ALLOW records, each with a unique rule/tool/agent name.

    Names are zero-padded (``r00`` < ``r01`` < ...) so every count is 1 and the
    deterministic tie-break (name ascending) makes the top-K lists exactly the
    first K names. This lets a per-list cap be asserted independently.
    """
    clock = _clock()
    return [
        AuditRecord(
            ts=clock(),
            call=ToolCall(tool_name=f"t{i:02d}", agent_id=f"a{i:02d}"),
            decision=Decision(verdict=Verdict.ALLOW, rule_id=f"r{i:02d}"),
        )
        for i in range(n)
    ]


class TestPerListTopOverridesSummary:
    def test_per_list_caps_apply_independently(self) -> None:
        lines = summarize_audit(
            _distinct_records(8), top_n=5, top_rules=2, top_tools=3, top_agents=4
        )
        text = "\n".join(lines)
        assert "top rules (by hits, max 2):" in text
        assert "top tools (by hits, max 3):" in text
        assert "top agents (by hits, max 4):" in text
        assert sum(f"  r{i:02d}" in text for i in range(8)) == 2
        assert sum(f"  t{i:02d}" in text for i in range(8)) == 3
        assert sum(f"  a{i:02d}" in text for i in range(8)) == 4

    def test_override_one_list_leaves_others_on_top_n(self) -> None:
        lines = summarize_audit(_distinct_records(8), top_n=3, top_agents=6)
        text = "\n".join(lines)
        assert "top rules (by hits, max 3):" in text
        assert "top tools (by hits, max 3):" in text
        assert "top agents (by hits, max 6):" in text
        assert sum(f"  r{i:02d}" in text for i in range(8)) == 3
        assert sum(f"  t{i:02d}" in text for i in range(8)) == 3
        assert sum(f"  a{i:02d}" in text for i in range(8)) == 6

    def test_omitting_all_three_is_byte_for_byte_unchanged(self) -> None:
        recs = _distinct_records(8)
        assert summarize_audit(recs, top_n=4) == summarize_audit(
            recs, top_n=4, top_rules=None, top_tools=None, top_agents=None
        )


class TestPerListTopOverridesDict:
    def test_per_list_caps_apply_independently(self) -> None:
        d = audit_stats_dict(
            _distinct_records(8), top_n=5, top_rules=2, top_tools=3, top_agents=4
        )
        assert len(d["top_rules"]) == 2
        assert len(d["top_tools"]) == 3
        assert len(d["top_agents"]) == 4

    def test_override_falls_back_to_top_n_when_omitted(self) -> None:
        d = audit_stats_dict(_distinct_records(8), top_n=3, top_tools=7)
        assert len(d["top_rules"]) == 3
        assert len(d["top_tools"]) == 7
        assert len(d["top_agents"]) == 3

    def test_omitting_all_three_is_unchanged(self) -> None:
        recs = _distinct_records(8)
        assert audit_stats_dict(recs, top_n=4) == audit_stats_dict(
            recs, top_n=4, top_rules=None, top_tools=None, top_agents=None
        )


class TestPerListTopOverridesCli:
    def _log(self, tmp_path: Path) -> Path:
        log = tmp_path / "many.jsonl"
        writer = JsonlAuditWriter(log, clock=_clock())
        with writer:
            for i in range(8):
                writer(
                    ToolCall(tool_name=f"t{i:02d}", agent_id=f"a{i:02d}"),
                    Decision(verdict=Verdict.ALLOW, rule_id=f"r{i:02d}"),
                )
        return log

    def test_text_top_with_agent_override(self, tmp_path: Path) -> None:
        log = self._log(tmp_path)
        rc, out, _ = _run(
            ["audit", "stats", str(log), "--top", "3", "--top-agents", "10"]
        )
        assert rc == 0
        assert sum(f"  r{i:02d}" in out for i in range(8)) == 3
        assert sum(f"  t{i:02d}" in out for i in range(8)) == 3
        assert sum(f"  a{i:02d}" in out for i in range(8)) == 8  # capped at 10, 8 exist

    def test_json_top_with_agent_override(self, tmp_path: Path) -> None:
        log = self._log(tmp_path)
        rc, out, _ = _run(
            ["audit", "stats", str(log), "--json", "--top", "3", "--top-agents", "10"]
        )
        assert rc == 0
        d = json.loads(out)
        assert len(d["top_rules"]) == 3
        assert len(d["top_tools"]) == 3
        assert len(d["top_agents"]) == 8


# --- R39: --fail-over CI gate -------------------------------------------------


def _flagged_log(path: Path) -> Path:
    """A 4-record log: 2 allow, 1 deny, 1 review (deny+review share = 50.0%)."""
    return _write_log(
        path,
        [
            ("read_file", Verdict.ALLOW, None),
            ("read_file", Verdict.ALLOW, None),
            ("send_email", Verdict.DENY, "no-exfil"),
            ("post_msg", Verdict.REVIEW, "needs-review"),
        ],
    )


class TestAuditFlaggedShare:
    def test_empty_log_is_zero(self) -> None:
        assert audit_flagged_share([]) == 0.0

    def test_share_is_exact_unrounded(self, tmp_path: Path) -> None:
        # 1 deny out of 3 records => 33.333...%, not the printed 33.3.
        log = _write_log(
            tmp_path / "x.jsonl",
            [
                ("a", Verdict.ALLOW, None),
                ("b", Verdict.ALLOW, None),
                ("c", Verdict.DENY, "r"),
            ],
        )
        share = audit_flagged_share(list(read_audit(str(log))))
        assert abs(share - 100.0 / 3.0) < 1e-9

    def test_matches_summary(self, tmp_path: Path) -> None:
        log = _flagged_log(tmp_path / "x.jsonl")
        assert audit_flagged_share(list(read_audit(str(log)))) == 50.0


class TestFailOverGate:
    def test_default_no_flag_exits_zero(self, tmp_path: Path) -> None:
        log = _flagged_log(tmp_path / "x.jsonl")
        rc, out, _ = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert "deny+review: 2/4" in out

    def test_over_threshold_exits_5_and_prints_summary(
        self, tmp_path: Path
    ) -> None:
        log = _flagged_log(tmp_path / "x.jsonl")
        rc, out, _ = _run(["audit", "stats", str(log), "--fail-over", "40"])
        assert rc == 5
        assert "deny+review: 2/4" in out  # summary still printed

    def test_at_threshold_exits_zero(self, tmp_path: Path) -> None:
        log = _flagged_log(tmp_path / "x.jsonl")
        rc, _, _ = _run(["audit", "stats", str(log), "--fail-over", "50"])
        assert rc == 0

    def test_under_threshold_exits_zero(self, tmp_path: Path) -> None:
        log = _flagged_log(tmp_path / "x.jsonl")
        rc, _, _ = _run(["audit", "stats", str(log), "--fail-over", "60"])
        assert rc == 0

    def test_exact_share_beats_rounded_print(self, tmp_path: Path) -> None:
        # 1/3 = 33.333% prints as 33.3 but must still trip a 33.3 threshold.
        log = _write_log(
            tmp_path / "x.jsonl",
            [
                ("a", Verdict.ALLOW, None),
                ("b", Verdict.ALLOW, None),
                ("c", Verdict.DENY, "r"),
            ],
        )
        rc, out, _ = _run(["audit", "stats", str(log), "--fail-over", "33.3"])
        assert "(33.3%)" in out
        assert rc == 5

    def test_empty_log_never_over_threshold(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        rc, out, _ = _run(["audit", "stats", str(empty), "--fail-over", "0"])
        assert rc == 0
        assert "records:     0" in out

    def test_json_branch_is_also_gated(self, tmp_path: Path) -> None:
        log = _flagged_log(tmp_path / "x.jsonl")
        rc, out, _ = _run(
            ["audit", "stats", str(log), "--json", "--fail-over", "40"]
        )
        assert rc == 5
        assert json.loads(out)["deny_review"]["count"] == 2

    def test_composes_with_verdict_filter(self, tmp_path: Path) -> None:
        log = _flagged_log(tmp_path / "x.jsonl")
        # Restrict to allow-only: flagged share drops to 0, gate passes.
        rc, _, _ = _run(
            ["audit", "stats", str(log), "--verdict", "allow", "--fail-over", "0"]
        )
        assert rc == 0
        # Restrict to deny-only: 100% flagged, even a high threshold trips.
        rc, _, _ = _run(
            ["audit", "stats", str(log), "--verdict", "deny", "--fail-over", "99"]
        )
        assert rc == 5


# --- audit_stats_csv (pure) + --csv CLI path (R40) ----------------------------


def _flagged_csv_log(path: Path) -> Path:
    """Log with 2 allow, 2 deny, 1 review (mirrors _mixed_records shares)."""
    return _write_log(
        path,
        [
            ("send_email", Verdict.DENY, "deny-web-to-email"),
            ("send_email", Verdict.DENY, "deny-web-to-email"),
            ("web_fetch", Verdict.ALLOW, None),
            ("kb_lookup", Verdict.ALLOW, "allow-internal-readers"),
            ("send_email", Verdict.REVIEW, "review-pii-egress"),
        ],
    )


class TestAuditStatsCsv:
    def test_empty_log_yields_only_header(self) -> None:
        assert audit_stats_csv([]) == ["verdict,count,pct"]

    def test_source_does_not_appear_in_body(self) -> None:
        # source is accepted for parity but never rendered into the CSV.
        assert audit_stats_csv([], source="x.jsonl") == ["verdict,count,pct"]

    def test_header_then_one_row_per_verdict_in_enum_order(self) -> None:
        lines = audit_stats_csv(_mixed_records())
        assert lines[0] == "verdict,count,pct"
        verdict_cells = [line.split(",")[0] for line in lines[1:-1]]
        assert verdict_cells == [v.value for v in Verdict]
        assert verdict_cells == ["allow", "deny", "review", "redact"]

    def test_counts_and_percentages_match_json_renderer(self) -> None:
        recs = _mixed_records()
        lines = audit_stats_csv(recs)
        rows = {
            line.split(",")[0]: line.split(",")[1:] for line in lines[1:]
        }
        assert rows["allow"] == ["2", "40.0"]
        assert rows["deny"] == ["2", "40.0"]
        assert rows["review"] == ["1", "20.0"]
        assert rows["redact"] == ["0", "0.0"]

    def test_trailing_deny_review_row(self) -> None:
        lines = audit_stats_csv(_mixed_records())
        assert lines[-1] == "deny+review,3,60.0"

    def test_rows_match_audit_stats_dict(self) -> None:
        recs = _mixed_records()
        d = audit_stats_dict(recs)
        lines = audit_stats_csv(recs)
        for verdict in Verdict:
            cells = next(
                line for line in lines if line.startswith(f"{verdict.value},")
            ).split(",")
            assert int(cells[1]) == d["verdicts"][verdict.value]["count"]
            assert float(cells[2]) == d["verdicts"][verdict.value]["pct"]
        dr = lines[-1].split(",")
        assert int(dr[1]) == d["deny_review"]["count"]
        assert float(dr[2]) == d["deny_review"]["pct"]

    def test_fields_have_no_commas_or_quotes(self) -> None:
        # Every data row is exactly three comma-separated cells (valid CSV
        # without quoting because no field embeds a comma).
        for line in audit_stats_csv(_mixed_records()):
            assert len(line.split(",")) == 3


class TestAuditStatsCsvCli:
    def test_csv_flag_emits_csv(self, tmp_path: Path) -> None:
        log = _flagged_csv_log(tmp_path / "a.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--csv"])
        assert rc == 0
        assert err == ""
        lines = out.splitlines()
        assert lines[0] == "verdict,count,pct"
        assert lines[1].startswith("allow,2,")
        assert lines[-1] == "deny+review,3,60.0"

    def test_csv_empty_log_prints_only_header(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.jsonl"
        log.write_text("", encoding="utf-8")
        rc, out, _ = _run(["audit", "stats", str(log), "--csv"])
        assert rc == 0
        assert out.splitlines() == ["verdict,count,pct"]

    def test_csv_and_json_together_is_argparse_error_exit_2(
        self, tmp_path: Path
    ) -> None:
        log = _flagged_csv_log(tmp_path / "a.jsonl")
        with pytest.raises(SystemExit) as exc:
            _run(["audit", "stats", str(log), "--csv", "--json"])
        assert exc.value.code == 2

    def test_csv_missing_file_exits_2(self, tmp_path: Path) -> None:
        rc, out, err = _run(
            ["audit", "stats", str(tmp_path / "nope.jsonl"), "--csv"]
        )
        assert rc == 2
        assert out == ""

    def test_csv_composes_with_fail_over_gate(self, tmp_path: Path) -> None:
        log = _flagged_csv_log(tmp_path / "a.jsonl")
        # flagged share is 60%, so a 40% threshold trips the gate (exit 5)
        rc, out, _ = _run(
            ["audit", "stats", str(log), "--csv", "--fail-over", "40"]
        )
        assert rc == 5
        # the CSV is still printed before the gate fires
        assert out.splitlines()[-1] == "deny+review,3,60.0"


# --- R41: rule-id filter (``apg audit stats --rule <glob>``) ------------------


class TestFilterByRule:
    """Pure ``filter_by_rule`` helper: fnmatch globs over ``decision.rule_id``."""

    def _records(self) -> list[AuditRecord]:
        # rule ids include the default/no-rule (None) bucket.
        rules: list[str | None] = ["deny-egress", "deny-secrets", "allow-kb", None]
        return [
            AuditRecord(
                ts="2026-06-08T00:00:00.000000Z",
                call=ToolCall(tool_name="t"),
                decision=Decision(verdict=Verdict.ALLOW, rule_id=rid),
            )
            for rid in rules
        ]

    def test_no_filter_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_rule(recs, None) == recs

    def test_empty_patterns_returns_all_unchanged(self) -> None:
        recs = self._records()
        assert filter_by_rule(recs, []) == recs

    def test_single_glob_matches_prefix(self) -> None:
        recs = self._records()
        out = filter_by_rule(recs, ["deny-*"])
        assert [r.decision.rule_id for r in out] == ["deny-egress", "deny-secrets"]

    def test_literal_pattern_is_exact_match(self) -> None:
        recs = self._records()
        out = filter_by_rule(recs, ["allow-kb"])
        assert [r.decision.rule_id for r in out] == ["allow-kb"]

    def test_multi_pattern_union(self) -> None:
        recs = self._records()
        out = filter_by_rule(recs, ["allow-kb", "deny-secrets"])
        assert [r.decision.rule_id for r in out] == ["deny-secrets", "allow-kb"]

    def test_default_bucket_selected_by_sentinel(self) -> None:
        recs = self._records()
        out = filter_by_rule(recs, [_NO_RULE])
        assert [r.decision.rule_id for r in out] == [None]

    def test_sentinel_composes_with_named_rule(self) -> None:
        recs = self._records()
        out = filter_by_rule(recs, ["allow-kb", _NO_RULE])
        assert [r.decision.rule_id for r in out] == ["allow-kb", None]

    def test_no_match_is_empty(self) -> None:
        recs = self._records()
        assert filter_by_rule(recs, ["nope-*"]) == []

    def test_matching_is_case_sensitive(self) -> None:
        recs = self._records()
        assert filter_by_rule(recs, ["DENY-*"]) == []

    def test_preserves_order(self) -> None:
        recs = self._records()
        out = filter_by_rule(recs, ["*"])
        # "*" matches every named rule; the None bucket maps to the sentinel
        # label, which "*" also matches, so all records pass through unchanged.
        assert out == recs


class TestAuditStatsRuleCli:
    """``apg audit stats --rule`` over a real JSONL log."""

    def _log(self, path: Path) -> Path:
        rows: list[tuple[str, Verdict, str | None]] = [
            ("send_email", Verdict.DENY, "deny-egress"),
            ("web_fetch", Verdict.ALLOW, None),
            ("kb_lookup", Verdict.ALLOW, "allow-kb"),
            ("send_email", Verdict.REVIEW, "deny-egress"),
        ]
        writer = JsonlAuditWriter(path, clock=_clock())
        with writer:
            for tool, verdict, rule in rows:
                writer(
                    ToolCall(tool_name=tool, agent_id="svc.x"),
                    Decision(verdict=verdict, rule_id=rule),
                )
        return path

    def test_glob_filters_text(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--rule", "deny-*"])
        assert rc == 0
        assert "records:     2" in out

    def test_glob_filters_json(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--json", "--rule", "deny-egress"]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 2
        assert data["verdicts"]["deny"]["count"] == 1
        assert data["verdicts"]["review"]["count"] == 1

    def test_repeatable_rule_unions(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--rule", "allow-kb", "--rule", "deny-egress",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 3

    def test_sentinel_selects_default_bucket(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(
            ["audit", "stats", str(log), "--json", "--rule", _NO_RULE]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 1
        assert data["verdicts"]["allow"]["count"] == 1

    def test_composes_with_verdict(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        # deny-egress has the two send_email rows; --verdict deny narrows to one.
        rc, out, err = _run(
            [
                "audit", "stats", str(log), "--json",
                "--rule", "deny-*", "--verdict", "deny",
            ]
        )
        assert rc == 0
        data = json.loads(out)
        assert data["records"] == 1
        assert data["verdicts"]["deny"]["count"] == 1

    def test_no_match_is_empty_log(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log), "--rule", "nope-*"])
        assert rc == 0
        assert "records:     0" in out
        assert "(log is empty - no records to summarize)" in out

    def test_no_rule_matches_all(self, tmp_path: Path) -> None:
        log = self._log(tmp_path / "t.jsonl")
        rc, out, err = _run(["audit", "stats", str(log)])
        assert rc == 0
        assert "records:     4" in out
