"""Tests for the TraceSig audit-trace export (R62).

Four layers, mirroring the module's contract:

* the event *shape* — the fixed key set, sorted label lists, and derived
  booleans a Sigma rule may reference without an existence guard;
* **losslessness** — ``event_to_record(record_to_event(r, seq)) == r``
  across a battery covering every optional field group;
* the ``apg audit export`` CLI (stdout/file/stdin, exit codes);
* the committed example traces under ``examples/traces/`` — regenerated
  in-process and compared byte-for-byte, so fixtures cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_policy_gateway.audit import AuditRecord, JsonlAuditWriter
from agent_policy_gateway.cli import main
from agent_policy_gateway.core import (
    Decision,
    Provenance,
    ProvenanceEntry,
    TaintLabel,
    ToolCall,
    Verdict,
)
from agent_policy_gateway.tracesig_export import (
    TRACESIG_SCHEMA,
    TRACESIG_SCHEMA_VERSION,
    TraceSigFormatError,
    event_to_record,
    export_events,
    read_tracesig,
    record_to_event,
    write_tracesig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = REPO_ROOT / "examples" / "traces"
SCHEMA_DOC = REPO_ROOT / "docs" / "tracesig-export.md"


# --------------------------------------------------------------------------- #
# Record builders                                                             #
# --------------------------------------------------------------------------- #


def _minimal_record() -> AuditRecord:
    """A legacy-shaped record: no optional field group present."""
    return AuditRecord(
        ts="2026-09-21T00:00:00.000000Z",
        call=ToolCall(tool_name="kb_lookup", args={"query": "x"}),
        decision=Decision(verdict=Verdict.ALLOW, reason="default-allow"),
    )


def _rich_record() -> AuditRecord:
    """A record exercising every optional field group at once."""
    prov = Provenance(
        (
            ProvenanceEntry(source="web", tool_name="web_fetch", call_id="c1"),
            ProvenanceEntry(source="pii", tool_name="crm_read", call_id=None),
        )
    )
    label = TaintLabel.of_dimensions(
        both=["web"], confidentiality=["pii"], integrity=["eval:untrusted"]
    )
    call = ToolCall(
        tool_name="send_email",
        args={"to": "a@b.example", "n": 3, "flag": True},
        input_label=label,
        agent_id="research-agent",
        call_id="c9",
        input_provenance=prov,
        arg_labels={"to": TaintLabel.of("web"), "body": label},
    )
    decision = Decision(
        verdict=Verdict.REDACT,
        rule_id="redact-pii",
        reason="masking",
        output_label=label.without(["pii"]),
        redacted_fields=("body",),
        output_provenance=prov.restrict_to(frozenset({"web"})),
        declassified_by=("grant-1", "grant-2"),
    )
    return AuditRecord(
        ts="2026-09-21T00:00:01.000000Z",
        call=call,
        decision=decision,
        prev="ab" * 32,
    )


_BATTERY = [
    _minimal_record(),
    _rich_record(),
    AuditRecord(
        ts="2026-09-21T00:00:02.000000Z",
        call=ToolCall(
            tool_name="send_money",
            input_label=TaintLabel.of("eval:untrusted"),
            agent_id="a",
            call_id="c2",
        ),
        decision=Decision(
            verdict=Verdict.DENY,
            rule_id="deny-untrusted",
            reason="nope",
            output_label=TaintLabel.of("eval:untrusted"),
        ),
        prev="0" * 64,
    ),
    AuditRecord(
        ts="2026-09-21T00:00:03.000000Z",
        call=ToolCall(tool_name="http_post", input_label=TaintLabel.of("pii")),
        decision=Decision(
            verdict=Verdict.REVIEW, rule_id="review-pii", reason="human loop"
        ),
    ),
]


# --------------------------------------------------------------------------- #
# Event shape                                                                 #
# --------------------------------------------------------------------------- #

#: Keys every event must carry, whatever the record looked like.
ALWAYS_PRESENT = {
    "schema",
    "schema_version",
    "seq",
    "ts",
    "tool",
    "agent",
    "call_id",
    "args",
    "verdict",
    "rule",
    "reason",
    "input_sources",
    "input_confidentiality",
    "input_integrity",
    "output_sources",
    "output_confidentiality",
    "output_integrity",
    "input_untrusted",
    "input_secret",
    "flagged",
}

#: Keys that appear only when the audit record carried them.
OPTIONAL_KEYS = {
    "redacted_fields",
    "declassified_by",
    "input_provenance",
    "output_provenance",
    "arg_labels",
    "prev",
}


class TestEventShape:
    def test_minimal_record_has_exactly_the_fixed_keys(self) -> None:
        event = record_to_event(_minimal_record(), 0)
        assert set(event) == ALWAYS_PRESENT

    def test_rich_record_adds_every_optional_key(self) -> None:
        event = record_to_event(_rich_record(), 7)
        assert set(event) == ALWAYS_PRESENT | OPTIONAL_KEYS
        assert event["seq"] == 7

    def test_schema_stamp(self) -> None:
        event = record_to_event(_minimal_record(), 0)
        assert event["schema"] == TRACESIG_SCHEMA == "apg-audit-trace"
        assert event["schema_version"] == TRACESIG_SCHEMA_VERSION == 1

    def test_label_lists_are_stored_canonical_sets_sorted(self) -> None:
        event = record_to_event(_rich_record(), 0)
        assert event["input_sources"] == ["web"]
        assert event["input_confidentiality"] == ["pii"]
        assert event["input_integrity"] == ["eval:untrusted"]
        # Output label had pii stripped in every dimension.
        assert event["output_sources"] == ["web"]
        assert event["output_confidentiality"] == []
        assert event["output_integrity"] == ["eval:untrusted"]

    def test_derived_booleans(self) -> None:
        rich = record_to_event(_rich_record(), 0)
        assert rich["input_untrusted"] is True  # web + eval:untrusted
        assert rich["input_secret"] is True  # web + pii
        assert rich["flagged"] is False  # redact is not deny/review
        deny = record_to_event(_BATTERY[2], 0)
        assert deny["flagged"] is True
        review = record_to_event(_BATTERY[3], 0)
        assert review["flagged"] is True
        clean = record_to_event(_minimal_record(), 0)
        assert clean["input_untrusted"] is False
        assert clean["input_secret"] is False

    def test_provenance_entry_shape(self) -> None:
        event = record_to_event(_rich_record(), 0)
        assert event["input_provenance"] == [
            {"source": "web", "tool": "web_fetch", "call_id": "c1"},
            {"source": "pii", "tool": "crm_read", "call_id": None},
        ]
        assert event["output_provenance"] == [
            {"source": "web", "tool": "web_fetch", "call_id": "c1"}
        ]

    def test_arg_labels_shape(self) -> None:
        event = record_to_event(_rich_record(), 0)
        assert event["arg_labels"]["to"] == {
            "sources": ["web"],
            "confidentiality": [],
            "integrity": [],
        }
        assert event["arg_labels"]["body"]["confidentiality"] == ["pii"]

    def test_event_is_json_serializable(self) -> None:
        for record in _BATTERY:
            json.dumps(record_to_event(record, 0))


# --------------------------------------------------------------------------- #
# Round trip                                                                  #
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    @pytest.mark.parametrize("record", _BATTERY, ids=lambda r: r.call.tool_name)
    def test_lossless(self, record: AuditRecord) -> None:
        assert event_to_record(record_to_event(record, 3)) == record

    def test_round_trip_through_json(self) -> None:
        """The trip survives an actual serialize/parse cycle, not just dicts."""
        for record in _BATTERY:
            line = json.dumps(record_to_event(record, 0), sort_keys=True)
            assert event_to_record(json.loads(line)) == record

    def test_derived_fields_are_ignored_on_import(self) -> None:
        event = record_to_event(_BATTERY[2], 0)
        event["flagged"] = False
        event["input_untrusted"] = False
        event["seq"] = 999
        assert event_to_record(event) == _BATTERY[2]

    def test_unknown_schema_rejected(self) -> None:
        event = record_to_event(_minimal_record(), 0)
        event["schema"] = "not-a-trace"
        with pytest.raises(TraceSigFormatError, match="unsupported schema"):
            event_to_record(event)

    def test_unknown_version_rejected(self) -> None:
        event = record_to_event(_minimal_record(), 0)
        event["schema_version"] = TRACESIG_SCHEMA_VERSION + 1
        with pytest.raises(TraceSigFormatError, match="schema_version"):
            event_to_record(event)

    def test_missing_required_key_rejected(self) -> None:
        event = record_to_event(_minimal_record(), 0)
        del event["tool"]
        with pytest.raises(TraceSigFormatError, match="missing required"):
            event_to_record(event)

    def test_missing_label_field_rejected(self) -> None:
        event = record_to_event(_minimal_record(), 0)
        del event["input_integrity"]
        with pytest.raises(TraceSigFormatError, match="input_integrity"):
            event_to_record(event)

    def test_unknown_verdict_rejected(self) -> None:
        event = record_to_event(_minimal_record(), 0)
        event["verdict"] = "shrug"
        with pytest.raises(TraceSigFormatError, match="unknown verdict"):
            event_to_record(event)


# --------------------------------------------------------------------------- #
# Streams and files                                                           #
# --------------------------------------------------------------------------- #


class TestStream:
    def test_export_events_counts_seq_in_order(self) -> None:
        events = list(export_events(_BATTERY))
        assert [e["seq"] for e in events] == list(range(len(_BATTERY)))
        assert [e["tool"] for e in events] == [r.call.tool_name for r in _BATTERY]

    def test_write_then_read_round_trips(self, tmp_path: Path) -> None:
        out = tmp_path / "trace.jsonl"
        with open(out, "w", encoding="utf-8") as fp:
            count = write_tracesig(_BATTERY, fp)
        assert count == len(_BATTERY)
        assert list(read_tracesig(out)) == _BATTERY

    def test_read_skips_blank_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "trace.jsonl"
        with open(out, "w", encoding="utf-8") as fp:
            write_tracesig(_BATTERY[:1], fp)
            fp.write("\n\n")
            write_tracesig(_BATTERY[1:2], fp)
        assert list(read_tracesig(out)) == _BATTERY[:2]

    def test_read_annotates_malformed_line(self, tmp_path: Path) -> None:
        out = tmp_path / "trace.jsonl"
        with open(out, "w", encoding="utf-8") as fp:
            write_tracesig(_BATTERY[:1], fp)
            fp.write("{not json\n")
        with pytest.raises(TraceSigFormatError, match="line 2"):
            list(read_tracesig(out))

    def test_read_rejects_non_object_line(self, tmp_path: Path) -> None:
        out = tmp_path / "trace.jsonl"
        out.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(TraceSigFormatError, match="expected object"):
            list(read_tracesig(out))


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def _write_audit_log(path: Path) -> None:
    clock_calls = iter(f"2026-09-21T00:00:0{i}.000000Z" for i in range(10))
    with JsonlAuditWriter(path, clock=lambda: next(clock_calls)) as writer:
        for record in _BATTERY:
            writer(record.call, record.decision)


class TestCli:
    def test_export_to_stdout(self, tmp_path: Path, capsys) -> None:
        log = tmp_path / "audit.jsonl"
        _write_audit_log(log)
        assert main(["audit", "export", str(log)]) == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == len(_BATTERY)
        events = [json.loads(line) for line in out]
        assert all(e["schema"] == TRACESIG_SCHEMA for e in events)
        assert [e["seq"] for e in events] == list(range(len(_BATTERY)))

    def test_export_to_file(self, tmp_path: Path, capsys) -> None:
        log = tmp_path / "audit.jsonl"
        _write_audit_log(log)
        out_file = tmp_path / "trace.jsonl"
        assert main(["audit", "export", str(log), "-o", str(out_file)]) == 0
        captured = capsys.readouterr()
        assert captured.out == ""  # stdout stays clean for piping
        assert f"wrote {len(_BATTERY)} event(s)" in captured.err
        events = [
            json.loads(line)
            for line in out_file.read_text(encoding="utf-8").splitlines()
        ]
        assert len(events) == len(_BATTERY)

    def test_export_explicit_format_flag(self, tmp_path: Path, capsys) -> None:
        log = tmp_path / "audit.jsonl"
        _write_audit_log(log)
        assert main(["audit", "export", str(log), "--format", "tracesig"]) == 0
        assert capsys.readouterr().out.count("\n") == len(_BATTERY)

    def test_export_reads_stdin(self, tmp_path: Path, capsys, monkeypatch) -> None:
        log = tmp_path / "audit.jsonl"
        _write_audit_log(log)
        with open(log, encoding="utf-8") as fp:
            monkeypatch.setattr("sys.stdin", fp)
            assert main(["audit", "export", "-"]) == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == len(_BATTERY)

    def test_missing_log_exits_2(self, tmp_path: Path, capsys) -> None:
        assert main(["audit", "export", str(tmp_path / "nope.jsonl")]) == 2
        assert "not found" in capsys.readouterr().err

    def test_malformed_log_exits_3(self, tmp_path: Path, capsys) -> None:
        log = tmp_path / "audit.jsonl"
        log.write_text("{broken\n", encoding="utf-8")
        assert main(["audit", "export", str(log)]) == 3
        assert "line 1" in capsys.readouterr().err

    def test_malformed_log_writes_no_output_file(
        self, tmp_path: Path, capsys
    ) -> None:
        log = tmp_path / "audit.jsonl"
        log.write_text("{broken\n", encoding="utf-8")
        out_file = tmp_path / "trace.jsonl"
        assert main(["audit", "export", str(log), "-o", str(out_file)]) == 3
        assert not out_file.exists()

    def test_unwritable_output_exits_2(self, tmp_path: Path, capsys) -> None:
        log = tmp_path / "audit.jsonl"
        _write_audit_log(log)
        target = tmp_path / "no-such-dir" / "trace.jsonl"
        assert main(["audit", "export", str(log), "-o", str(target)]) == 2
        assert "cannot write" in capsys.readouterr().err

    def test_unknown_format_is_a_usage_error(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["audit", "export", "x.jsonl", "--format", "csv"])
        assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# Committed example traces                                                    #
# --------------------------------------------------------------------------- #


class TestExampleTraces:
    def test_committed_fixtures_regenerate_byte_identical(
        self, tmp_path: Path
    ) -> None:
        from examples.traces import SCENARIOS, generate_traces

        generated = generate_traces(tmp_path)
        assert set(generated) == set(SCENARIOS)
        for audit_path, tracesig_path in generated.values():
            for fresh in (audit_path, tracesig_path):
                committed = TRACES_DIR / fresh.name
                assert committed.exists(), f"missing committed fixture {fresh.name}"
                assert fresh.read_bytes() == committed.read_bytes(), (
                    f"{fresh.name} drifted from the committed fixture; "
                    "regenerate with `python -m examples.traces`"
                )

    def test_committed_export_matches_committed_audit_log(self) -> None:
        from agent_policy_gateway.audit import read_audit
        from examples.traces import SCENARIOS

        for name in SCENARIOS:
            records = list(read_audit(TRACES_DIR / f"{name}.audit.jsonl"))
            exported = list(
                read_tracesig(TRACES_DIR / f"{name}.tracesig.jsonl")
            )
            assert exported == records, name

    def test_narrative_invariants_hold(self, tmp_path: Path) -> None:
        from examples.traces import expectations_hold, generate_traces

        generated = generate_traces(tmp_path)
        failures = [
            claim for claim, ok in expectations_hold(generated) if not ok
        ]
        assert failures == []

    def test_every_committed_event_is_stamped(self) -> None:
        from examples.traces import SCENARIOS

        for name in SCENARIOS:
            path = TRACES_DIR / f"{name}.tracesig.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                assert event["schema"] == TRACESIG_SCHEMA
                assert event["schema_version"] == TRACESIG_SCHEMA_VERSION
                assert ALWAYS_PRESENT <= set(event)


# --------------------------------------------------------------------------- #
# Docs lockstep                                                               #
# --------------------------------------------------------------------------- #


class TestSchemaDoc:
    def test_doc_exists_and_names_the_frozen_schema(self) -> None:
        text = SCHEMA_DOC.read_text(encoding="utf-8")
        assert TRACESIG_SCHEMA in text
        assert f"**{TRACESIG_SCHEMA_VERSION}**" in text

    def test_doc_documents_every_event_field(self) -> None:
        text = SCHEMA_DOC.read_text(encoding="utf-8")
        for field in sorted(ALWAYS_PRESENT | OPTIONAL_KEYS):
            assert f"`{field}`" in text, f"docs/tracesig-export.md misses {field}"

    def test_doc_names_the_cli_command_and_regeneration(self) -> None:
        text = SCHEMA_DOC.read_text(encoding="utf-8")
        assert "apg audit export" in text
        assert "python -m examples.traces" in text
