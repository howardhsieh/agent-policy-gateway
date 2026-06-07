"""Tests for agent_policy_gateway.audit (R5).

These cover the JSONL writer (append semantics, parent-dir creation,
gateway integration on allow + deny paths, fsync flag, context manager,
closed-state error), the read-back path (round-trip, blank-line skip,
malformed-JSON line numbers, missing-key detection), and the
``apg-replay`` CLI (timeline content, --verdict / --limit filters,
missing-file exit code, malformed-log exit code).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_policy_gateway import (
    GENESIS_PREV,
    AuditFormatError,
    AuditRecord,
    ChainVerifyResult,
    Decision,
    Gateway,
    JsonlAuditWriter,
    PolicyDenied,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    format_record,
    load_policy_str,
    read_audit,
    replay_main,
    verify_chain,
)

# --- helpers ---------------------------------------------------------------


def _ts_clock(values: list[str]) -> Iterator[str]:
    """Return a callable that yields successive timestamps from ``values``."""
    it = iter(values)

    def next_ts() -> str:
        return next(it)

    return next_ts


def _make_call(
    tool: str = "send_email",
    *,
    args: dict | None = None,
    sources: tuple[str, ...] = (),
    agent_id: str | None = "agent.research",
    call_id: str | None = "c-1",
) -> ToolCall:
    return ToolCall(
        tool_name=tool,
        args=args or {"to": "ops@example.com", "body": "hi"},
        input_label=TaintLabel.of(*sources),
        agent_id=agent_id,
        call_id=call_id,
    )


def _make_decision(
    verdict: Verdict = Verdict.ALLOW,
    *,
    rule_id: str | None = "allow-default",
    reason: str = "",
    sources: tuple[str, ...] = (),
) -> Decision:
    return Decision(
        verdict=verdict,
        rule_id=rule_id,
        reason=reason,
        output_label=TaintLabel.of(*sources),
    )


# --- AuditRecord round-trip -------------------------------------------------


def test_audit_record_round_trips_through_dict() -> None:
    rec = AuditRecord(
        ts="2026-05-01T00:00:00.000000Z",
        call=_make_call(sources=("web",)),
        decision=_make_decision(sources=("web",)),
    )
    d = rec.to_dict()
    assert AuditRecord.from_dict(d) == rec


def test_audit_record_from_dict_rejects_missing_keys() -> None:
    with pytest.raises(AuditFormatError) as ei:
        AuditRecord.from_dict({"ts": "x", "call": {}})
    assert "decision" in str(ei.value)


# --- JsonlAuditWriter -------------------------------------------------------


def test_writer_writes_one_line_per_call(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    clock = _ts_clock(["2026-05-01T00:00:00.000000Z", "2026-05-01T00:00:01.000000Z"])
    writer = JsonlAuditWriter(log, clock=clock)
    try:
        writer(_make_call(call_id="a"), _make_decision())
        writer(_make_call(call_id="b"), _make_decision(verdict=Verdict.DENY))
    finally:
        writer.close()

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["call"]["call_id"] == "a"
    assert first["decision"]["verdict"] == "allow"
    assert second["call"]["call_id"] == "b"
    assert second["decision"]["verdict"] == "deny"
    assert first["ts"] == "2026-05-01T00:00:00.000000Z"


def test_writer_creates_missing_parent_directory(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "deep" / "audit.jsonl"
    assert not log.parent.exists()
    with JsonlAuditWriter(log) as writer:
        writer(_make_call(), _make_decision())
    assert log.exists()
    assert log.read_text(encoding="utf-8").count("\n") == 1


def test_writer_is_append_only_across_opens(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    with JsonlAuditWriter(log, clock=lambda: "T1") as w:
        w(_make_call(call_id="a"), _make_decision())
    with JsonlAuditWriter(log, clock=lambda: "T2") as w:
        w(_make_call(call_id="b"), _make_decision())

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["call"]["call_id"] == "a"
    assert json.loads(lines[1])["call"]["call_id"] == "b"


def test_writer_records_are_one_line_each(tmp_path: Path) -> None:
    """JSONL invariant: no embedded newlines."""
    log = tmp_path / "audit.jsonl"
    multiline_args = {"body": "line1\nline2\nline3"}
    with JsonlAuditWriter(log) as w:
        w(_make_call(args=multiline_args), _make_decision())
    text = log.read_text(encoding="utf-8")
    # Exactly one trailing newline, no others.
    assert text.count("\n") == 1
    record = json.loads(text)
    assert record["call"]["args"]["body"] == "line1\nline2\nline3"


def test_writer_after_close_raises(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    writer = JsonlAuditWriter(log)
    writer.close()
    assert writer.closed is True
    with pytest.raises(ValueError, match="closed"):
        writer(_make_call(), _make_decision())


def test_writer_close_is_idempotent(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    writer = JsonlAuditWriter(log)
    writer.close()
    writer.close()  # must not raise
    assert writer.closed is True


def test_writer_fsync_flag_is_accepted(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    with JsonlAuditWriter(log, fsync=True) as w:
        w(_make_call(), _make_decision())
    # The flag exercised the fsync codepath; the on-disk content is what we
    # really care about.
    assert log.read_text(encoding="utf-8").endswith("\n")


def test_writer_path_property(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    writer = JsonlAuditWriter(log)
    try:
        assert writer.path == str(log)
    finally:
        writer.close()


# --- read_audit -------------------------------------------------------------


def test_read_audit_yields_records_in_file_order(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    clock = _ts_clock([f"T{i}" for i in range(1, 4)])  # T1, T2, T3
    with JsonlAuditWriter(log, clock=clock) as w:
        for i in range(3):
            w(_make_call(call_id=f"c{i}"), _make_decision(rule_id=f"r{i}"))

    records = list(read_audit(log))
    assert [r.ts for r in records] == ["T1", "T2", "T3"]
    assert [r.call.call_id for r in records] == ["c0", "c1", "c2"]
    assert [r.decision.rule_id for r in records] == ["r0", "r1", "r2"]


def test_read_audit_skips_blank_lines(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    record = JsonlAuditWriter.build_record(_make_call(), _make_decision(), ts="T1")
    line = json.dumps(record, sort_keys=True)
    log.write_text(f"\n{line}\n\n{line}\n   \n", encoding="utf-8")
    records = list(read_audit(log))
    assert len(records) == 2


def test_read_audit_raises_on_malformed_json(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    record = JsonlAuditWriter.build_record(_make_call(), _make_decision(), ts="T1")
    log.write_text(json.dumps(record) + "\n{not json\n", encoding="utf-8")
    it = read_audit(log)
    next(it)  # first line is fine
    with pytest.raises(AuditFormatError) as ei:
        next(it)
    assert "line 2" in str(ei.value)


def test_read_audit_raises_on_non_object_line(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(AuditFormatError, match="line 1"):
        list(read_audit(log))


def test_read_audit_raises_on_missing_field(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text(json.dumps({"ts": "T1", "call": {}}) + "\n", encoding="utf-8")
    with pytest.raises(AuditFormatError, match="line 1"):
        list(read_audit(log))


# --- format_record ----------------------------------------------------------


def test_format_record_includes_key_fields() -> None:
    rec = AuditRecord(
        ts="2026-05-01T00:00:00.000000Z",
        call=_make_call(sources=("web",)),
        decision=_make_decision(
            verdict=Verdict.DENY,
            rule_id="block-exfiltration",
            reason="web-tainted output sent to external recipient",
            sources=("web",),
        ),
    )
    out = format_record(rec)
    assert "DENY" in out
    assert "send_email" in out
    assert "agent=agent.research" in out
    assert "rule=block-exfiltration" in out
    assert "reason: web-tainted output sent to external recipient" in out
    assert "['web']" in out  # both input + output labels print as sorted lists


def test_format_record_omits_empty_optional_fields() -> None:
    rec = AuditRecord(
        ts="T1",
        call=ToolCall(tool_name="lookup"),
        decision=Decision(verdict=Verdict.ALLOW),
    )
    out = format_record(rec)
    assert "ALLOW" in out
    assert "lookup" in out
    # No reason / labels / args / agent → those lines must not appear.
    assert "reason:" not in out
    assert "input:" not in out
    assert "output:" not in out
    assert "args:" not in out
    assert "agent=" not in out


def test_format_record_truncates_long_args() -> None:
    rec = AuditRecord(
        ts="T1",
        call=_make_call(args={"body": "x" * 500}),
        decision=_make_decision(),
    )
    out = format_record(rec)
    args_line = next(line for line in out.splitlines() if line.lstrip().startswith("args:"))
    assert "..." in args_line
    assert len(args_line) < 500


# --- replay CLI -------------------------------------------------------------


def _populate(log: Path, n_allow: int = 1, n_deny: int = 1) -> None:
    clock = _ts_clock([f"2026-05-01T00:00:0{i}.000000Z" for i in range(n_allow + n_deny + 5)])
    with JsonlAuditWriter(log, clock=clock) as w:
        for i in range(n_allow):
            w(
                _make_call(call_id=f"a{i}"),
                _make_decision(rule_id="allow-default"),
            )
        for i in range(n_deny):
            w(
                _make_call(call_id=f"d{i}", sources=("web",)),
                _make_decision(
                    verdict=Verdict.DENY,
                    rule_id="block-exfiltration",
                    reason="exfiltration",
                    sources=("web",),
                ),
            )


def test_replay_main_prints_human_readable_timeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "audit.jsonl"
    _populate(log, n_allow=1, n_deny=1)

    rc = replay_main([str(log)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ALLOW" in captured.out
    assert "DENY" in captured.out
    assert "send_email" in captured.out
    assert "rule=block-exfiltration" in captured.out


def test_replay_main_filters_by_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "audit.jsonl"
    _populate(log, n_allow=2, n_deny=1)

    rc = replay_main([str(log), "--verdict", "deny"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "DENY" in captured.out
    assert "ALLOW" not in captured.out


def test_replay_main_limits_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "audit.jsonl"
    _populate(log, n_allow=3, n_deny=2)

    rc = replay_main([str(log), "--limit", "2"])
    captured = capsys.readouterr()
    # Each record begins with a "[..." header line; count those.
    headers = [line for line in captured.out.splitlines() if line.startswith("[")]
    assert rc == 0
    assert len(headers) == 2


def test_replay_main_returns_2_on_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = replay_main([str(tmp_path / "nope.jsonl")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "log not found" in captured.err


def test_replay_main_returns_3_on_malformed_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text("not json\n", encoding="utf-8")
    rc = replay_main([str(log)])
    captured = capsys.readouterr()
    assert rc == 3
    assert "apg-replay" in captured.err
    assert "line 1" in captured.err


# --- gateway integration ----------------------------------------------------


_DEFAULT_POLICY_YAML = """\
name: test-default
rules:
  - id: block-exfiltration
    when:
      tool: "send_email"
      taint:
        any_of: ["web"]
    effect:
      action: deny
      reason: "web-tainted send_email blocked"
  - id: allow-default
    when:
      tool: "*"
    effect:
      action: allow
"""


def _gateway(audit_writer: JsonlAuditWriter) -> Gateway:
    gw = Gateway(
        policies=[load_policy_str(_DEFAULT_POLICY_YAML)],
        audit_writer=audit_writer,
    )
    gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))
    gw.register_tool("send_email", ToolTaintSpec.of())
    return gw


def test_writer_records_allowed_call_through_gateway(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    with JsonlAuditWriter(log, clock=lambda: "T1") as audit:
        gw = _gateway(audit)

        @gw.wrap_tool(tool_name="send_email", taint_spec=ToolTaintSpec.of())
        def send_email(to: str, body: str) -> dict:
            return {"to": to}

        send_email("ops@example.com", "hi")

    records = list(read_audit(log))
    assert len(records) == 1
    assert records[0].decision.verdict is Verdict.ALLOW
    assert records[0].call.tool_name == "send_email"
    assert records[0].decision.rule_id == "allow-default"


def test_writer_records_denied_call_through_gateway(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    with JsonlAuditWriter(log, clock=lambda: "T1") as audit:
        gw = _gateway(audit)

        @gw.wrap_tool(tool_name="send_email", taint_spec=ToolTaintSpec.of())
        def send_email(to: str, body: str) -> dict:  # noqa: ARG001 - never reached
            raise AssertionError("must not be called when policy denies")

        with pytest.raises(PolicyDenied):
            send_email(
                "ops@example.com",
                "hi",
                apg_input_label=TaintLabel.of("web"),
            )

    records = list(read_audit(log))
    assert len(records) == 1
    assert records[0].decision.verdict is Verdict.DENY
    assert records[0].decision.rule_id == "block-exfiltration"
    assert "web" in records[0].call.input_label.sources


def test_writer_round_trips_full_exfiltration_scenario(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: write a real allow + deny pair, replay them as text."""
    log = tmp_path / "audit.jsonl"
    times = iter(["T1", "T2"])
    with JsonlAuditWriter(log, clock=lambda: next(times)) as audit:
        gw = _gateway(audit)

        @gw.wrap_tool(tool_name="web_search", taint_spec=ToolTaintSpec.of(adds=("web",)))
        def web_search(query: str) -> str:  # noqa: ARG001
            return "snippet"

        @gw.wrap_tool(tool_name="send_email", taint_spec=ToolTaintSpec.of())
        def send_email(to: str, body: str) -> dict:  # noqa: ARG001
            return {"to": to}

        web_search("apg")
        with pytest.raises(PolicyDenied):
            send_email(
                "ops@example.com",
                "hi",
                apg_input_label=TaintLabel.of("web"),
            )

    rc = replay_main([str(log)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ALLOW" in captured.out
    assert "DENY" in captured.out
    assert "web_search" in captured.out
    assert "send_email" in captured.out


# --- hash-chained audit log (R27) -------------------------------------------


def _chained_log(tmp_path: Path, n: int = 3) -> Path:
    """Write ``n`` chained records and return the log path."""
    log = tmp_path / "chained.jsonl"
    clock = _ts_clock([f"2026-06-07T00:00:0{i}.000000Z" for i in range(n + 2)])
    with JsonlAuditWriter(log, chain=True, clock=clock) as w:
        for i in range(n):
            w(_make_call(call_id=f"c{i}"), _make_decision())
    return log


def test_chain_disabled_by_default_has_no_prev_field(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    with JsonlAuditWriter(log) as w:
        w(_make_call(), _make_decision())
    record = json.loads(log.read_text(encoding="utf-8"))
    assert "prev" not in record


def test_chain_writes_prev_field_and_genesis_sentinel(tmp_path: Path) -> None:
    log = _chained_log(tmp_path, n=3)
    lines = log.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert first["prev"] == GENESIS_PREV
    # Each later record's prev is the sha256 of the previous serialized line.
    import hashlib

    for i in range(1, len(lines)):
        expected = hashlib.sha256(lines[i - 1].encode("utf-8")).hexdigest()
        assert json.loads(lines[i])["prev"] == expected


def test_chained_records_round_trip_through_read_audit(tmp_path: Path) -> None:
    log = _chained_log(tmp_path, n=3)
    records = list(read_audit(log))
    assert len(records) == 3
    assert records[0].prev == GENESIS_PREV
    assert all(r.prev is not None for r in records)
    # to_dict/from_dict preserves prev.
    assert AuditRecord.from_dict(records[1].to_dict()).prev == records[1].prev


def test_chain_continues_across_reopens(tmp_path: Path) -> None:
    log = tmp_path / "chained.jsonl"
    with JsonlAuditWriter(log, chain=True, clock=lambda: "T1") as w:
        w(_make_call(call_id="a"), _make_decision())
    with JsonlAuditWriter(log, chain=True, clock=lambda: "T2") as w:
        w(_make_call(call_id="b"), _make_decision())
    result = verify_chain(log)
    assert result.ok is True
    assert result.records == 2


def test_verify_chain_accepts_intact_log(tmp_path: Path) -> None:
    log = _chained_log(tmp_path, n=4)
    result = verify_chain(log)
    assert isinstance(result, ChainVerifyResult)
    assert result.ok is True
    assert result.records == 4
    assert result.broken_line is None


def test_verify_chain_detects_in_place_edit(tmp_path: Path) -> None:
    log = _chained_log(tmp_path, n=3)
    lines = log.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["decision"]["verdict"] = "deny"  # silently flip allow -> deny
    lines[1] = json.dumps(tampered, ensure_ascii=False, sort_keys=True)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = verify_chain(log)
    assert result.ok is False
    # Editing line 2 breaks line 3's prev link.
    assert result.broken_line == 3


def test_verify_chain_detects_deleted_record(tmp_path: Path) -> None:
    log = _chained_log(tmp_path, n=3)
    lines = log.read_text(encoding="utf-8").splitlines()
    del lines[1]  # drop the middle record
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = verify_chain(log)
    assert result.ok is False
    assert result.broken_line == 2


def test_verify_chain_detects_mid_line_truncation(tmp_path: Path) -> None:
    log = _chained_log(tmp_path, n=3)
    text = log.read_text(encoding="utf-8")
    # Chop the file partway through the last line.
    log.write_text(text[: len(text) - 30], encoding="utf-8")
    result = verify_chain(log)
    assert result.ok is False
    assert result.broken_line == 3


def test_verify_chain_flags_unchained_log(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    with JsonlAuditWriter(log) as w:  # chain=False -> no prev field
        w(_make_call(), _make_decision())
    result = verify_chain(log)
    assert result.ok is False
    assert result.broken_line == 1
    assert "prev" in (result.reason or "")


def test_replay_verify_exits_0_on_intact_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _chained_log(tmp_path, n=3)
    rc = replay_main([str(log), "--verify"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "chain intact" in captured.out


def test_replay_verify_exits_4_on_tampered_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _chained_log(tmp_path, n=3)
    lines = log.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["ts"] = "1999-01-01T00:00:00.000000Z"
    lines[1] = json.dumps(tampered, ensure_ascii=False, sort_keys=True)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc = replay_main([str(log), "--verify"])
    captured = capsys.readouterr()
    assert rc == 4
    assert "chain broken at line 3" in captured.err


def test_replay_verify_exits_2_on_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = replay_main([str(tmp_path / "nope.jsonl"), "--verify"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "log not found" in captured.err


def test_unchained_log_still_replays_normally(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "audit.jsonl"
    _populate(log, n_allow=1, n_deny=1)
    rc = replay_main([str(log)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ALLOW" in captured.out
    assert "DENY" in captured.out
