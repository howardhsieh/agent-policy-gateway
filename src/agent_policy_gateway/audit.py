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
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import IO, Any

from agent_policy_gateway.core import Decision, ToolCall, Verdict

__all__ = [
    "GENESIS_PREV",
    "AuditFormatError",
    "AuditRecord",
    "ChainVerifyResult",
    "JsonlAuditWriter",
    "format_record",
    "read_audit",
    "replay_main",
    "verify_chain",
]


# Fixed sentinel stored in the ``prev`` field of a chain's first (genesis)
# record. SHA-256 digests are 64 lowercase hex chars, so an all-zero string of
# the same width is unambiguous and never collides with a real digest.
GENESIS_PREV = "0" * 64


def _line_digest(line: str) -> str:
    """SHA-256 hex digest of one serialized record line (newline excluded)."""
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


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
    prev: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ts": self.ts,
            "call": self.call.to_dict(),
            "decision": self.decision.to_dict(),
        }
        # Serialized only when present so legacy (unchained) records keep
        # their exact prior on-disk shape and round-trip unchanged.
        if self.prev is not None:
            d["prev"] = self.prev
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditRecord:
        missing = {"ts", "call", "decision"} - set(d)
        if missing:
            raise AuditFormatError(
                f"audit record missing required key(s): {sorted(missing)}"
            )
        prev = d.get("prev")
        return cls(
            ts=str(d["ts"]),
            call=ToolCall.from_dict(d["call"]),
            decision=Decision.from_dict(d["decision"]),
            prev=None if prev is None else str(prev),
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
        chain: When True, every written record carries a ``prev`` field
            holding the SHA-256 digest of the previous record's serialized
            line (the first record uses :data:`GENESIS_PREV`). This makes
            truncation, deletion, and in-place edits detectable via
            :func:`verify_chain` / ``apg-replay --verify``. Default False
            so the legacy record shape is preserved. When re-opening an
            existing chained log the running digest is seeded from the last
            line so the chain continues unbroken.
        clock: Override for the timestamp source -- handy in tests so
            records are deterministic.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fsync: bool = False,
        chain: bool = False,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._path = os.fspath(path)
        self._fsync = bool(fsync)
        self._chain = bool(chain)
        self._clock = clock or _utc_now_iso
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Seed the running digest before opening for append so a re-opened
        # chained log continues from its last line rather than restarting.
        self._prev_digest = (
            self._last_line_digest() if self._chain else None
        )
        self._fp: IO[str] | None = open(self._path, "a", encoding="utf-8")

    def _last_line_digest(self) -> str:
        """Digest of the last non-blank line already on disk, or the genesis
        sentinel when the log is absent or empty."""
        try:
            with open(self._path, encoding="utf-8") as fp:
                last = ""
                for raw in fp:
                    stripped = raw.strip()
                    if stripped:
                        last = stripped
        except FileNotFoundError:
            return GENESIS_PREV
        return _line_digest(last) if last else GENESIS_PREV

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
        record = self.build_record(
            call, decision, ts=self._clock(), prev=self._prev_digest
        )
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        self._fp.write(line + "\n")
        self._fp.flush()
        if self._fsync:
            os.fsync(self._fp.fileno())
        if self._chain:
            # The next record's ``prev`` is the digest of the line just written.
            self._prev_digest = _line_digest(line)

    @staticmethod
    def build_record(
        call: ToolCall,
        decision: Decision,
        *,
        ts: str | None = None,
        prev: str | None = None,
    ) -> dict[str, Any]:
        """Produce the dict form of one record without touching the file."""
        return AuditRecord(
            ts=ts or _utc_now_iso(),
            call=call,
            decision=decision,
            prev=prev,
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


# --- chain verification (R27) -------------------------------------------------


@dataclass(frozen=True)
class ChainVerifyResult:
    """Outcome of walking an audit log's hash chain.

    Attributes:
        ok: True when every record's ``prev`` matched the running digest.
        records: Number of non-blank records examined.
        broken_line: 1-based file line number of the first broken/anomalous
            record, or ``None`` when ``ok``.
        reason: Human-readable description of the first break, or ``None``.
    """

    ok: bool
    records: int
    broken_line: int | None = None
    reason: str | None = None


def verify_chain(path: str | os.PathLike[str]) -> ChainVerifyResult:
    """Walk a JSONL audit log and verify its ``prev`` hash chain.

    The first record's ``prev`` must equal :data:`GENESIS_PREV`; every
    subsequent record's ``prev`` must equal the SHA-256 digest of the
    immediately preceding record's serialized line. Any in-place edit,
    deleted record, or mid-line truncation breaks a link and is reported
    with the offending 1-based file line number.

    Raises:
        FileNotFoundError: if ``path`` does not exist (callers map this to
            their own exit code, mirroring :func:`read_audit`).
    """
    expected = GENESIS_PREV
    records = 0
    with open(os.fspath(path), encoding="utf-8") as fp:
        for lineno, raw in enumerate(fp, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                return ChainVerifyResult(
                    ok=False,
                    records=records,
                    broken_line=lineno,
                    reason=f"invalid JSON: {exc.msg}",
                )
            if not isinstance(data, dict):
                return ChainVerifyResult(
                    ok=False,
                    records=records,
                    broken_line=lineno,
                    reason=f"expected object, got {type(data).__name__}",
                )
            prev = data.get("prev")
            if prev is None:
                return ChainVerifyResult(
                    ok=False,
                    records=records,
                    broken_line=lineno,
                    reason="record has no 'prev' field (log not hash-chained?)",
                )
            if prev != expected:
                return ChainVerifyResult(
                    ok=False,
                    records=records,
                    broken_line=lineno,
                    reason="prev digest does not match previous record",
                )
            records += 1
            expected = _line_digest(line)
    return ChainVerifyResult(ok=True, records=records)


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
    if not dec.output_provenance.is_empty():
        origins = ", ".join(
            f"{e.source}<-{e.tool_name}@{e.call_id or '?'}"
            for e in dec.output_provenance.entries
        )
        lines.append(f"  origin: {origins}")
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
        log line is malformed. With ``--verify``: ``0`` if the hash chain
        is intact, ``2`` if the file is missing, ``4`` if the chain is
        broken (the first broken line number is printed to stderr).
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Verify the tamper-evident hash chain instead of printing the "
            "timeline. Exits 0 if intact, 4 if a link is broken."
        ),
    )
    args = parser.parse_args(argv)

    if args.verify:
        try:
            result = verify_chain(args.log)
        except FileNotFoundError:
            print(f"apg-replay: log not found: {args.log}", file=sys.stderr)
            return 2
        if result.ok:
            print(f"apg-replay: chain intact ({result.records} records)")
            return 0
        print(
            f"apg-replay: chain broken at line {result.broken_line}: "
            f"{result.reason}",
            file=sys.stderr,
        )
        return 4

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
