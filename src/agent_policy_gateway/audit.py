"""Append-only JSONL audit log + replay tool (R5).

The gateway in :mod:`agent_policy_gateway.gateway` accepts any callable
matching :data:`AuditWriter` and invokes it once per decision *before*
the underlying tool runs (fail-closed-on-audit). R4 only specified the
interface; R5 ships a concrete on-disk implementation and a tool to
read the resulting log back.

Two pieces:

* :class:`JsonlAuditWriter` -- a callable class that opens a file in
  append mode and appends one JSON object per ``(call, decision)`` pair.
  Each record is a single line so the file is trivially seekable,
  greppable, and tail-able. Writes are flushed after every line so a
  crash leaves a recoverable log; pass ``fsync=True`` for durability
  through power loss.
* :func:`read_audit` and :func:`replay_main` -- the read side. The CLI
  ``apg-replay LOG`` reads a JSONL file and prints a human-readable
  timeline. Filters: ``--verdict {allow,deny,review}`` and
  ``--limit N``.

The on-disk record schema is::

    {
      "ts": "<iso-8601 utc>",
      "call":     { ToolCall.to_dict() },
      "decision": { Decision.to_dict() }
    }

Records round-trip through :class:`AuditRecord.to_dict` /
:meth:`AuditRecord.from_dict` so callers can reuse the dict form for
non-JSONL sinks (databases, queues) without touching this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import IO, Any

from agent_policy_gateway.core import Decision, ToolCall, Verdict

__all__ = [
    "AuditFormatError",
    "AuditRecord",
    "JsonlAuditWriter",
    "format_record",
    "read_audit",
    "replay_main",
]


class AuditFormatError(ValueError):
    """Raised when an audit log line cannot be parsed.

    The exception message includes the line number (1-based) so the
    caller can point at the offending row.
    """


@dataclass(frozen=True)
class AuditRecord:
    """One entry in an audit log: a timestamp, the call, and the decision."""

    ts: str
    call: ToolCall
    decision: Decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "call": self.call.to_dict(),
            "decision": self.decision.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditRecord:
        missing = {"ts", "call", "decision"} - set(d)
        if missing:
            raise AuditFormatError(
                f"audit record missing required key(s): {sorted(missing)}"
            )
        return cls(
            ts=str(d["ts"]),
            call=ToolCall.from_dict(d["call"]),
            decision=Decision.from_dict(d["decision"]),
        )


def _utc_now_iso() -> str:
    """Return current UTC time as ``YYYY-MM-DDTHH:MM:SS.ffffffZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class JsonlAuditWriter:
    """Append-only JSONL writer compatible with the gateway's ``AuditWriter``.

    Each instance owns a file handle opened with ``mode="a"``, which
    maps to ``O_APPEND`` on POSIX. Writes shorter than ``PIPE_BUF`` (at
    least 4 KiB on Linux) are atomic across processes, so multiple
    gateways pointed at the same log file will not interleave records
    line-by-line on typical record sizes.

    Usage::

        with JsonlAuditWriter("audit.jsonl") as audit:
            gateway = Gateway(policies=[...], audit_writer=audit)
            ...

    The writer is also a plain callable, so existing :class:`Gateway`
    instances built with ``audit_writer=writer`` keep working without
    changes.

    Args:
        path: Destination file. Parent directories are created if
            missing.
        fsync: When True, ``os.fsync`` is called after each write.
            Default False (flush only) for performance.
        clock: Override for the timestamp source -- handy in tests so
            records are deterministic.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fsync: bool = False,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._path = os.fspath(path)
        self._fsync = bool(fsync)
        self._clock = clock or _utc_now_iso
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fp: IO[str] | None = open(self._path, "a", encoding="utf-8")

    @property
    def path(self) -> str:
        """Path the writer is appending to."""
        return self._path

    @property
    def closed(self) -> bool:
        return self._fp is None or self._fp.closed

    def __call__(self, call: ToolCall, decision: Decision) -> None:
        if self._fp is None or self._fp.closed:
            raise ValueError("audit writer is closed")
        record = self.build_record(call, decision, ts=self._clock())
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        self._fp.write(line + "\n")
        self._fp.flush()
        if self._fsync:
            os.fsync(self._fp.fileno())

    @staticmethod
    def build_record(
        call: ToolCall, decision: Decision, *, ts: str | None = None
    ) -> dict[str, Any]:
        """Produce the dict form of one record without touching the file."""
        return AuditRecord(
            ts=ts or _utc_now_iso(),
            call=call,
            decision=decision,
        ).to_dict()

    def close(self) -> None:
        if self._fp is not None and not self._fp.closed:
            self._fp.close()
        self._fp = None

    def __enter__(self) -> JsonlAuditWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def read_audit(path: str | os.PathLike[str]) -> Iterator[AuditRecord]:
    """Yield :class:`AuditRecord` objects from a JSONL log in file order.

    The file is opened eagerly, so a missing path raises
    :class:`FileNotFoundError` at call time rather than on first
    iteration -- callers (notably :func:`replay_main`) rely on this to
    distinguish missing-file from malformed-content failures.

    Blank lines are skipped (so files written by this module's writer
    and then concatenated still parse). A line that fails to parse as
    JSON, or that parses as something other than the audit-record
    schema, raises :class:`AuditFormatError` annotated with the line
    number; the caller can decide whether to abort or continue.
    """
    fp = open(os.fspath(path), encoding="utf-8")
    return _iter_audit(fp)


def _iter_audit(fp: IO[str]) -> Iterator[AuditRecord]:
    try:
        for lineno, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditFormatError(
                    f"line {lineno}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise AuditFormatError(
                    f"line {lineno}: expected object, got {type(data).__name__}"
                )
            try:
                yield AuditRecord.from_dict(data)
            except AuditFormatError as exc:
                raise AuditFormatError(f"line {lineno}: {exc}") from exc
    finally:
        fp.close()


# --- replay CLI ---------------------------------------------------------------


def _truncate(s: str, limit: int = 200) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def format_record(record: AuditRecord) -> str:
    """Render one :class:`AuditRecord` as a multi-line, scan-friendly block.

    Empty optional fields are omitted so the timeline stays compact.
    """
    call = record.call
    dec = record.decision
    head = f"[{record.ts}] {dec.verdict.value.upper():<6s} {call.tool_name}"
    if call.agent_id:
        head += f"  agent={call.agent_id}"
    if dec.rule_id:
        head += f"  rule={dec.rule_id}"
    if call.call_id:
        head += f"  call_id={call.call_id}"
    lines = [head]
    if dec.reason:
        lines.append(f"  reason: {dec.reason}")
    if call.input_label.sources:
        lines.append(f"  input:  {sorted(call.input_label.sources)}")
    if dec.output_label.sources:
        lines.append(f"  output: {sorted(dec.output_label.sources)}")
    if call.args:
        args_json = json.dumps(
            call.args, ensure_ascii=False, sort_keys=True, default=str
        )
        lines.append(f"  args:   {_truncate(args_json)}")
    return "\n".join(lines)


def replay_main(argv: list[str] | None = None) -> int:
    """Entry point for ``apg-replay``.

    Returns:
        ``0`` on success, ``2`` if the log file is missing, ``3`` if a
        log line is malformed.
    """
    parser = argparse.ArgumentParser(
        prog="apg-replay",
        description=(
            "Replay an agent-policy-gateway JSONL audit log as a "
            "human-readable timeline."
        ),
    )
    parser.add_argument("log", help="Path to the JSONL audit log file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Print only the first N matching records.",
    )
    parser.add_argument(
        "--verdict",
        choices=[v.value for v in Verdict],
        default=None,
        help="Only print records with the given verdict.",
    )
    args = parser.parse_args(argv)

    try:
        records = read_audit(args.log)
    except FileNotFoundError:
        print(f"apg-replay: log not found: {args.log}", file=sys.stderr)
        return 2

    count = 0
    try:
        for record in records:
            if args.verdict is not None and record.decision.verdict.value != args.verdict:
                continue
            print(format_record(record))
            count += 1
            if args.limit is not None and count >= args.limit:
                break
    except AuditFormatError as exc:
        print(f"apg-replay: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(replay_main())
