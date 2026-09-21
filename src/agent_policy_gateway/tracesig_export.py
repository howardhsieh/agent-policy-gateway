"""Versioned audit-trace export for TraceSig (R62).

APG is the *prevention* half of a pair whose *detection* half is the
sibling TraceSig project: Sigma-style rules matched over agent tool-call
traces. This module freezes the trace format that pairing consumes — a
flat, Sigma-friendly JSONL event stream derived one-to-one from APG's
audit log — and provides the pure mapping in both directions.

Schema (frozen, versioned)
--------------------------

Every exported line is one JSON object ("event") for one
:class:`~agent_policy_gateway.audit.AuditRecord`, in log order. Two
fields identify the format so a TraceSig rule pack can pin what it was
written against:

* ``schema`` — always :data:`TRACESIG_SCHEMA` (``"apg-audit-trace"``).
* ``schema_version`` — :data:`TRACESIG_SCHEMA_VERSION`, an integer.
  The compatibility promise: within one major version, existing fields
  keep their name, type, and meaning; new *optional* fields may appear.
  Any change that renames, retypes, or repurposes a field bumps the
  version.

The remaining fields come in three groups (the full mapping table lives
in ``docs/tracesig-export.md``):

* **Always present** — ``seq`` (0-based position in the exported log),
  ``ts``, ``tool``, ``agent``, ``call_id``, ``args``, ``verdict``,
  ``rule``, ``reason``, the six per-side label lists
  (``input_sources`` / ``input_confidentiality`` / ``input_integrity``
  and the ``output_*`` triple, sorted, possibly empty), and the derived
  detection booleans ``input_untrusted`` (effective integrity set
  non-empty), ``input_secret`` (effective confidentiality set
  non-empty), and ``flagged`` (verdict is deny or review). A fixed key
  set means a Sigma rule can reference any of these without an
  existence guard.
* **Present only when carried** — ``redacted_fields``,
  ``declassified_by``, ``input_provenance`` / ``output_provenance``
  (lists of ``{source, tool, call_id}``), ``arg_labels`` (per-argument
  label triples, R57), and ``prev`` (the R27 hash-chain digest). Their
  absence means the audit record did not carry them, so a legacy log
  exports without invented fields.
* **Derived** — the three booleans above are computed *from* the label
  fields for rule-authoring convenience and are ignored (recomputed,
  never trusted) when an event is turned back into a record.

The mapping is **lossless**: :func:`event_to_record` inverts
:func:`record_to_event` exactly, which the round-trip test pins. Labels
are exported as their *stored* (already canonical) dimension sets — not
the effective per-dimension unions — precisely so reconstruction is
exact; detection rules that want the effective sets use the derived
booleans or union the lists themselves.

Pure module: no I/O beyond the explicit ``write_tracesig`` helper; the
``apg audit export`` subcommand in :mod:`agent_policy_gateway.cli`
drives it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import IO, Any

from agent_policy_gateway.audit import AuditRecord
from agent_policy_gateway.core import (
    Decision,
    Provenance,
    ProvenanceEntry,
    TaintLabel,
    ToolCall,
    Verdict,
)

__all__ = [
    "TRACESIG_SCHEMA",
    "TRACESIG_SCHEMA_VERSION",
    "TraceSigFormatError",
    "event_to_record",
    "export_events",
    "read_tracesig",
    "record_to_event",
    "write_tracesig",
]


#: The ``schema`` identifier stamped on every exported event.
TRACESIG_SCHEMA = "apg-audit-trace"

#: The frozen schema version stamped on every exported event. Within one
#: version, existing fields keep their name, type, and meaning; any
#: breaking change bumps this integer (see the module docstring).
TRACESIG_SCHEMA_VERSION = 1


class TraceSigFormatError(ValueError):
    """Raised when a trace event cannot be mapped back to an audit record."""


def _label_lists(label: TaintLabel) -> tuple[list[str], list[str], list[str]]:
    """The stored (canonical) dimension sets of ``label``, sorted."""
    return (
        sorted(label.sources),
        sorted(label.confidentiality),
        sorted(label.integrity),
    )


def _provenance_list(prov: Provenance) -> list[dict[str, Any]]:
    """Provenance entries as ``{source, tool, call_id}`` dicts, chain order."""
    return [
        {"source": e.source, "tool": e.tool_name, "call_id": e.call_id}
        for e in prov.entries
    ]


def record_to_event(record: AuditRecord, seq: int) -> dict[str, Any]:
    """Map one :class:`AuditRecord` to one TraceSig event dict.

    ``seq`` is the record's 0-based position in the exported log; TraceSig
    rules use it to order events within a trace file independent of
    timestamp ties. The returned dict is JSON-serializable as-is and
    inverts exactly through :func:`event_to_record`.
    """
    call = record.call
    dec = record.decision
    in_src, in_conf, in_integ = _label_lists(call.input_label)
    out_src, out_conf, out_integ = _label_lists(dec.output_label)
    event: dict[str, Any] = {
        "schema": TRACESIG_SCHEMA,
        "schema_version": TRACESIG_SCHEMA_VERSION,
        "seq": seq,
        "ts": record.ts,
        "tool": call.tool_name,
        "agent": call.agent_id,
        "call_id": call.call_id,
        "args": dict(call.args),
        "verdict": dec.verdict.value,
        "rule": dec.rule_id,
        "reason": dec.reason,
        "input_sources": in_src,
        "input_confidentiality": in_conf,
        "input_integrity": in_integ,
        "output_sources": out_src,
        "output_confidentiality": out_conf,
        "output_integrity": out_integ,
        # Derived detection conveniences; recomputed on import, never trusted.
        "input_untrusted": bool(call.input_label.integrity_sources),
        "input_secret": bool(call.input_label.confidentiality_sources),
        "flagged": dec.verdict in (Verdict.DENY, Verdict.REVIEW),
    }
    # Optional groups: emitted only when the audit record carried them, so a
    # legacy log exports without invented fields (mirroring the audit
    # writer's own only-when-present serialization).
    if dec.redacted_fields:
        event["redacted_fields"] = list(dec.redacted_fields)
    if dec.declassified_by:
        event["declassified_by"] = list(dec.declassified_by)
    if not call.input_provenance.is_empty():
        event["input_provenance"] = _provenance_list(call.input_provenance)
    if not dec.output_provenance.is_empty():
        event["output_provenance"] = _provenance_list(dec.output_provenance)
    if call.arg_labels:
        event["arg_labels"] = {
            name: {
                "sources": s,
                "confidentiality": c,
                "integrity": i,
            }
            for name, (s, c, i) in (
                (name, _label_lists(label))
                for name, label in call.arg_labels.items()
            )
        }
    if record.prev is not None:
        event["prev"] = record.prev
    return event


def _label_from_event(
    event: dict[str, Any], prefix: str, *, context: str
) -> TaintLabel:
    """Rebuild a :class:`TaintLabel` from an event's ``<prefix>_*`` lists."""
    try:
        return TaintLabel(
            sources=frozenset(event[f"{prefix}_sources"]),
            confidentiality=frozenset(event[f"{prefix}_confidentiality"]),
            integrity=frozenset(event[f"{prefix}_integrity"]),
        )
    except KeyError as exc:
        raise TraceSigFormatError(
            f"{context}: missing label field {exc.args[0]!r}"
        ) from None


def _provenance_from_event(entries: list[dict[str, Any]]) -> Provenance:
    return Provenance(
        tuple(
            ProvenanceEntry(
                source=str(e["source"]),
                tool_name=str(e["tool"]),
                call_id=e.get("call_id"),
            )
            for e in entries
        )
    )


def event_to_record(event: dict[str, Any]) -> AuditRecord:
    """Invert :func:`record_to_event`: rebuild the exact audit record.

    Only events carrying the supported :data:`TRACESIG_SCHEMA` /
    :data:`TRACESIG_SCHEMA_VERSION` stamp are accepted; anything else
    raises :class:`TraceSigFormatError` rather than guessing. The derived
    fields (``input_untrusted`` / ``input_secret`` / ``flagged`` /
    ``seq``) are ignored — they are recomputed views, not state.
    """
    schema = event.get("schema")
    if schema != TRACESIG_SCHEMA:
        raise TraceSigFormatError(
            f"unsupported schema {schema!r} (expected {TRACESIG_SCHEMA!r})"
        )
    version = event.get("schema_version")
    if version != TRACESIG_SCHEMA_VERSION:
        raise TraceSigFormatError(
            f"unsupported schema_version {version!r} "
            f"(this build reads version {TRACESIG_SCHEMA_VERSION})"
        )
    missing = {"ts", "tool", "verdict"} - set(event)
    if missing:
        raise TraceSigFormatError(
            f"event missing required key(s): {sorted(missing)}"
        )
    arg_labels = {
        name: TaintLabel(
            sources=frozenset(ld.get("sources", [])),
            confidentiality=frozenset(ld.get("confidentiality", [])),
            integrity=frozenset(ld.get("integrity", [])),
        )
        for name, ld in (event.get("arg_labels") or {}).items()
    }
    call = ToolCall(
        tool_name=str(event["tool"]),
        args=dict(event.get("args") or {}),
        input_label=_label_from_event(event, "input", context="call"),
        agent_id=event.get("agent"),
        call_id=event.get("call_id"),
        input_provenance=_provenance_from_event(
            event.get("input_provenance") or []
        ),
        arg_labels=arg_labels,
    )
    try:
        verdict = Verdict(event["verdict"])
    except ValueError as exc:
        raise TraceSigFormatError(f"unknown verdict: {event['verdict']!r}") from exc
    decision = Decision(
        verdict=verdict,
        rule_id=event.get("rule"),
        reason=str(event.get("reason") or ""),
        output_label=_label_from_event(event, "output", context="decision"),
        redacted_fields=tuple(event.get("redacted_fields") or ()),
        output_provenance=_provenance_from_event(
            event.get("output_provenance") or []
        ),
        declassified_by=tuple(event.get("declassified_by") or ()),
    )
    prev = event.get("prev")
    return AuditRecord(
        ts=str(event["ts"]),
        call=call,
        decision=decision,
        prev=None if prev is None else str(prev),
    )


def export_events(records: Iterable[AuditRecord]) -> Iterator[dict[str, Any]]:
    """Yield one TraceSig event per audit record, in order, ``seq`` counted."""
    for seq, record in enumerate(records):
        yield record_to_event(record, seq)


def write_tracesig(records: Iterable[AuditRecord], fp: IO[str]) -> int:
    """Write ``records`` to ``fp`` as TraceSig JSONL; return the event count.

    One event per line, keys sorted, no trailing spaces — the same stable
    serialization the audit writer uses, so committed fixtures diff
    cleanly.
    """
    count = 0
    for event in export_events(records):
        fp.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        count += 1
    return count


def read_tracesig(path: str | os.PathLike[str]) -> Iterator[AuditRecord]:
    """Read a TraceSig JSONL file back into :class:`AuditRecord` objects.

    The verification half of the round-trip (TraceSig itself only reads
    events; this reader exists so tests and tooling can prove the export
    lost nothing). Blank lines are skipped; a malformed line raises
    :class:`TraceSigFormatError` annotated with its 1-based line number.
    """
    with open(os.fspath(path), encoding="utf-8") as fp:
        for lineno, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceSigFormatError(
                    f"line {lineno}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise TraceSigFormatError(
                    f"line {lineno}: expected object, got {type(data).__name__}"
                )
            try:
                yield event_to_record(data)
            except TraceSigFormatError as exc:
                raise TraceSigFormatError(f"line {lineno}: {exc}") from None
