"""Tests for ``apg audit diff`` and the pure diff helpers (R47).

Two halves, mirroring ``test_audit_stats.py``: the pure
:func:`audit_diff_dict` / :func:`summarize_audit_diff` helpers are driven
directly (identical logs, added/removed rules, verdict deltas, empty sides),
and the CLI is driven through ``main(argv)`` so no subprocess is needed.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from agent_policy_gateway import audit_diff_dict, summarize_audit_diff
from agent_policy_gateway.audit import (
    _NO_AGENT,
    _NO_RULE,
    AuditRecord,
    JsonlAuditWriter,
    read_audit,
)
from agent_policy_gateway.cli import main
from agent_policy_gateway.core import Decision, ToolCall, Verdict

#: ``(tool, verdict, rule_id, agent_id)`` rows used to build synthetic logs.
Row = tuple[str, Verdict, str | None, str | None]


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
        return f"2026-08-13T00:{n // 60:02d}:{n % 60:02d}.000000Z"

    return tick


def _records(rows: list[Row]) -> list[AuditRecord]:
    """Build in-memory records without touching the filesystem."""
    clock = _clock()
    return [
        AuditRecord(
            ts=clock(),
            call=ToolCall(tool_name=tool, agent_id=agent),
            decision=Decision(verdict=verdict, rule_id=rule),
        )
        for tool, verdict, rule, agent in rows
    ]


def _write_log(path: Path, rows: list[Row]) -> Path:
    """Write ``rows`` to a JSONL audit log on disk."""
    with JsonlAuditWriter(path, clock=_clock()) as writer:
        for tool, verdict, rule, agent in rows:
            writer(
                ToolCall(tool_name=tool, agent_id=agent),
                Decision(verdict=verdict, rule_id=rule),
            )
    return path


OLD_ROWS: list[Row] = [
    ("send_email", Verdict.ALLOW, "r.allow", "a1"),
    ("send_email", Verdict.ALLOW, "r.allow", "a1"),
    ("read_file", Verdict.DENY, "r.deny", "a2"),
    ("read_file", Verdict.REVIEW, "r.review", "a2"),
]

NEW_ROWS: list[Row] = [
    ("send_email", Verdict.ALLOW, "r.allow", "a1"),
    ("read_file", Verdict.DENY, "r.deny", "a2"),
    ("read_file", Verdict.DENY, "r.deny", "a2"),
    ("http_get", Verdict.DENY, "r.net", "a3"),
    ("http_get", Verdict.DENY, "r.net", "a3"),
]


# --- audit_diff_dict (pure) ---------------------------------------------------


class TestAuditDiffDictRecords:
    def test_record_counts_and_delta(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        assert diff["records"] == {"old": 4, "new": 5, "delta": 1}

    def test_both_sides_empty(self) -> None:
        diff = audit_diff_dict([], [])
        assert diff["records"] == {"old": 0, "new": 0, "delta": 0}
        assert diff["rules"] == []
        assert diff["tools"] == []
        assert diff["agents"] == []
        for verdict in Verdict:
            block = diff["verdicts"][verdict.value]
            assert block["old"] == {"count": 0, "pct": 0.0}
            assert block["new"] == {"count": 0, "pct": 0.0}
            assert block["delta"] == {"count": 0, "pct": 0.0}

    def test_empty_old_side_is_all_additions(self) -> None:
        diff = audit_diff_dict([], _records(NEW_ROWS))
        assert diff["records"] == {"old": 0, "new": 5, "delta": 5}
        assert {e["status"] for e in diff["tools"]} == {"added"}
        assert all(e["old_rank"] is None for e in diff["tools"])
        assert all(e["rank_delta"] is None for e in diff["tools"])

    def test_empty_new_side_is_all_removals(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), [])
        assert diff["records"] == {"old": 4, "new": 0, "delta": -4}
        assert {e["status"] for e in diff["rules"]} == {"removed"}
        assert all(e["new_rank"] is None for e in diff["rules"])


class TestAuditDiffDictIdenticalLogs:
    def test_identical_logs_report_no_movement(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(OLD_ROWS))
        assert diff["records"]["delta"] == 0
        assert diff["rules"] == []
        assert diff["tools"] == []
        assert diff["agents"] == []

    def test_identical_logs_have_zero_verdict_deltas(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(OLD_ROWS))
        for verdict in Verdict:
            delta = diff["verdicts"][verdict.value]["delta"]
            assert delta["count"] == 0
            # Never negative zero, so the rendered line reads ``+0.0``.
            assert delta["pct"] == 0.0
            assert f"{delta['pct']:+.1f}" == "+0.0"
        assert diff["deny_review"]["delta"] == {"count": 0, "pct": 0.0}

    def test_scaled_but_proportional_log_moves_counts_not_shares(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(OLD_ROWS * 2))
        assert diff["records"]["delta"] == 4
        for verdict in Verdict:
            block = diff["verdicts"][verdict.value]
            assert block["delta"]["pct"] == 0.0
            assert block["old"]["pct"] == block["new"]["pct"]


class TestAuditDiffDictVerdicts:
    def test_verdict_counts_shares_and_deltas(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        allow = diff["verdicts"]["allow"]
        assert allow["old"] == {"count": 2, "pct": 50.0}
        assert allow["new"] == {"count": 1, "pct": 20.0}
        assert allow["delta"] == {"count": -1, "pct": -30.0}
        deny = diff["verdicts"]["deny"]
        assert deny["old"]["count"] == 1
        assert deny["new"]["count"] == 4
        assert deny["delta"] == {"count": 3, "pct": 55.0}

    def test_every_verdict_present_in_enum_order(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        assert list(diff["verdicts"]) == [v.value for v in Verdict]
        # ``redact`` never occurs in either log but is still reported.
        assert diff["verdicts"]["redact"]["delta"] == {"count": 0, "pct": 0.0}

    def test_deny_review_combines_the_flagged_verdicts(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        assert diff["deny_review"]["old"] == {"count": 2, "pct": 50.0}
        assert diff["deny_review"]["new"] == {"count": 4, "pct": 80.0}
        assert diff["deny_review"]["delta"] == {"count": 2, "pct": 30.0}

    def test_pct_delta_is_the_difference_of_the_printed_shares(self) -> None:
        # 1/3 -> 1/6 prints 33.3% -> 16.7%, so the delta must read -16.6, not
        # the exact -16.67 (see ``_delta_pct``).
        old = _records([("t", Verdict.ALLOW, "r", "a")] + [("t", Verdict.DENY, "r", "a")] * 2)
        new = _records([("t", Verdict.ALLOW, "r", "a")] + [("t", Verdict.DENY, "r", "a")] * 5)
        diff = audit_diff_dict(old, new)
        assert diff["verdicts"]["allow"]["old"]["pct"] == 33.3
        assert diff["verdicts"]["allow"]["new"]["pct"] == 16.7
        assert diff["verdicts"]["allow"]["delta"]["pct"] == -16.6


class TestAuditDiffDictMovement:
    def test_added_rule_is_reported_with_no_old_rank(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        added = [e for e in diff["rules"] if e["name"] == "r.net"]
        assert added == [
            {
                "name": "r.net",
                "status": "added",
                "old_count": 0,
                "new_count": 2,
                "delta": 2,
                "old_rank": None,
                "new_rank": 2,
                "rank_delta": None,
            }
        ]

    def test_removed_rule_is_reported_with_no_new_rank(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        removed = [e for e in diff["rules"] if e["name"] == "r.review"]
        assert removed == [
            {
                "name": "r.review",
                "status": "removed",
                "old_count": 1,
                "new_count": 0,
                "delta": -1,
                "old_rank": 3,
                "new_rank": None,
                "rank_delta": None,
            }
        ]

    def test_rank_delta_is_positive_when_an_entry_climbs(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        moved = {e["name"]: e for e in diff["rules"]}
        # r.deny goes from rank 2 to rank 1: it climbed one place.
        assert moved["r.deny"]["status"] == "moved"
        assert moved["r.deny"]["rank_delta"] == 1
        # r.allow slips from rank 1 to rank 3.
        assert moved["r.allow"]["rank_delta"] == -2

    def test_entry_with_unchanged_count_but_changed_rank_is_reported(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        moved = {e["name"]: e for e in diff["tools"]}
        assert moved["read_file"]["delta"] == 0
        assert moved["read_file"]["rank_delta"] == -1

    def test_additions_and_removals_sort_ahead_of_moves(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        statuses = [e["status"] for e in diff["rules"]]
        assert statuses[: statuses.count("moved") or None] != ["moved"]
        assert statuses.index("added") < statuses.index("moved")
        assert statuses.index("removed") < statuses.index("moved")

    def test_top_n_caps_each_axis_independently_of_the_others(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS), top_n=1)
        assert len(diff["rules"]) == 1
        assert len(diff["tools"]) == 1
        assert len(diff["agents"]) == 1

    def test_non_positive_top_n_reports_nothing(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS), top_n=0)
        assert diff["rules"] == []
        assert diff["tools"] == []
        assert diff["agents"] == []

    def test_unnamed_buckets_use_the_stats_sentinel_labels(self) -> None:
        old = _records([("t", Verdict.ALLOW, None, None)])
        new = _records([("t", Verdict.ALLOW, None, None)] * 3)
        diff = audit_diff_dict(old, new)
        assert [e["name"] for e in diff["rules"]] == [_NO_RULE]
        assert [e["name"] for e in diff["agents"]] == [_NO_AGENT]

    def test_agents_axis_tracks_agent_ids(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        assert [e["name"] for e in diff["agents"] if e["status"] == "added"] == ["a3"]

    def test_result_is_json_serializable(self) -> None:
        diff = audit_diff_dict(_records(OLD_ROWS), _records(NEW_ROWS))
        assert json.loads(json.dumps(diff)) == diff


# --- summarize_audit_diff (pure) ----------------------------------------------


class TestSummarizeAuditDiff:
    def test_header_names_both_logs(self) -> None:
        lines = summarize_audit_diff(
            _records(OLD_ROWS),
            _records(NEW_ROWS),
            old_source="old.jsonl",
            new_source="new.jsonl",
        )
        assert lines[0] == "audit log diff: old.jsonl -> new.jsonl"

    def test_header_omits_sources_when_both_absent(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS))
        assert lines[0] == "audit log diff"

    def test_record_line_carries_a_signed_delta(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS))
        assert lines[1] == "records:     4 -> 5  (+1)"

    def test_both_empty_logs_short_circuit(self) -> None:
        lines = summarize_audit_diff([], [])
        assert lines == [
            "audit log diff",
            "records:     0 -> 0  (+0)",
            "(both logs are empty - nothing to compare)",
        ]

    def test_every_verdict_gets_a_line_in_enum_order(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS))
        start = lines.index("verdicts:")
        rendered = [line.split()[0] for line in lines[start + 1 : start + 5]]
        assert rendered == [v.value for v in Verdict]

    def test_verdict_line_shows_counts_shares_and_signed_deltas(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS))
        allow = next(line for line in lines if line.strip().startswith("allow"))
        assert "2 -> 1" in allow
        assert "(-1)" in allow
        assert "50.0% -> 20.0%" in allow
        assert "(-30.0)" in allow

    def test_deny_review_line(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS))
        flagged = next(line for line in lines if line.startswith("deny+review:"))
        assert flagged.startswith("deny+review: 2/4 -> 4/5  (+2)")
        assert "50.0% -> 80.0% (+30.0)" in flagged

    def test_movement_sections_use_singular_labels(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS), top_n=3)
        for label in ("rule", "tool", "agent"):
            assert f"top {label} changes (by movement, max 3):" in lines

    def test_added_and_removed_entries_are_marked(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS))
        assert any(
            line.startswith("  + r.net: 0 -> 2 (+2)  rank - -> 2") for line in lines
        )
        assert any(
            line.startswith("  - r.review: 1 -> 0 (-1)  rank 3 -> -")
            for line in lines
        )

    def test_unchanged_axis_prints_no_change(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(OLD_ROWS))
        assert lines.count("  (no change)") == 3

    def test_top_n_caps_the_rendered_entries(self) -> None:
        lines = summarize_audit_diff(_records(OLD_ROWS), _records(NEW_ROWS), top_n=1)
        start = lines.index("top rule changes (by movement, max 1):")
        assert not lines[start + 2].startswith("  ")


# --- apg audit diff (CLI) -----------------------------------------------------


class TestAuditDiffCli:
    def test_text_output_and_exit_zero(self, tmp_path: Path) -> None:
        old = _write_log(tmp_path / "old.jsonl", OLD_ROWS)
        new = _write_log(tmp_path / "new.jsonl", NEW_ROWS)
        rc, out, err = _run(["audit", "diff", str(old), str(new)])
        assert rc == 0
        assert err == ""
        assert out.splitlines()[0] == f"audit log diff: {old} -> {new}"
        assert "records:     4 -> 5  (+1)" in out

    def test_identical_logs_still_exit_zero(self, tmp_path: Path) -> None:
        old = _write_log(tmp_path / "a.jsonl", OLD_ROWS)
        new = _write_log(tmp_path / "b.jsonl", OLD_ROWS)
        rc, out, _err = _run(["audit", "diff", str(old), str(new)])
        assert rc == 0
        assert out.count("  (no change)") == 3

    def test_json_output_echoes_both_paths(self, tmp_path: Path) -> None:
        old = _write_log(tmp_path / "old.jsonl", OLD_ROWS)
        new = _write_log(tmp_path / "new.jsonl", NEW_ROWS)
        rc, out, _err = _run(["audit", "diff", str(old), str(new), "--json"])
        assert rc == 0
        payload = json.loads(out)
        assert payload["old"] == str(old)
        assert payload["new"] == str(new)
        assert payload["records"] == {"old": 4, "new": 5, "delta": 1}

    def test_json_matches_the_pure_helper(self, tmp_path: Path) -> None:
        old = _write_log(tmp_path / "old.jsonl", OLD_ROWS)
        new = _write_log(tmp_path / "new.jsonl", NEW_ROWS)
        _rc, out, _err = _run(["audit", "diff", str(old), str(new), "--json"])
        payload = json.loads(out)
        expected = audit_diff_dict(list(read_audit(old)), list(read_audit(new)))
        assert {k: v for k, v in payload.items() if k not in ("old", "new")} == expected

    def test_top_flag_caps_each_axis(self, tmp_path: Path) -> None:
        old = _write_log(tmp_path / "old.jsonl", OLD_ROWS)
        new = _write_log(tmp_path / "new.jsonl", NEW_ROWS)
        _rc, out, _err = _run(
            ["audit", "diff", str(old), str(new), "--json", "--top", "1"]
        )
        payload = json.loads(out)
        assert len(payload["rules"]) == 1
        assert len(payload["tools"]) == 1
        assert len(payload["agents"]) == 1

    def test_missing_old_log_exits_two(self, tmp_path: Path) -> None:
        new = _write_log(tmp_path / "new.jsonl", NEW_ROWS)
        missing = tmp_path / "nope.jsonl"
        rc, out, err = _run(["audit", "diff", str(missing), str(new)])
        assert rc == 2
        assert out == ""
        assert "old audit log not found" in err

    def test_missing_new_log_exits_two(self, tmp_path: Path) -> None:
        old = _write_log(tmp_path / "old.jsonl", OLD_ROWS)
        missing = tmp_path / "nope.jsonl"
        rc, _out, err = _run(["audit", "diff", str(old), str(missing)])
        assert rc == 2
        assert "new audit log not found" in err

    def test_malformed_old_log_exits_three(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json}\n", encoding="utf-8")
        new = _write_log(tmp_path / "new.jsonl", NEW_ROWS)
        rc, _out, err = _run(["audit", "diff", str(bad), str(new)])
        assert rc == 3
        assert "old log: line 1" in err

    def test_malformed_new_log_exits_three(self, tmp_path: Path) -> None:
        old = _write_log(tmp_path / "old.jsonl", OLD_ROWS)
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"ts": "x"}\n', encoding="utf-8")
        rc, _out, err = _run(["audit", "diff", str(old), str(bad)])
        assert rc == 3
        assert "new log: line 1" in err

    def test_empty_logs_are_reported_not_an_error(self, tmp_path: Path) -> None:
        old = tmp_path / "old.jsonl"
        old.write_text("", encoding="utf-8")
        new = tmp_path / "new.jsonl"
        new.write_text("", encoding="utf-8")
        rc, out, _err = _run(["audit", "diff", str(old), str(new)])
        assert rc == 0
        assert "(both logs are empty - nothing to compare)" in out
