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
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
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
    "audit_stats_dict",
    "filter_by_verdict",
    "format_record",
    "read_audit",
    "read_audit_stdin",
    "replay_main",
    "summarize_audit",
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


def read_audit_stdin() -> Iterator[AuditRecord]:
    """Yield :class:`AuditRecord` objects parsed from ``sys.stdin``.

    The streaming counterpart to :func:`read_audit`: instead of opening a path
    it reads the process's standard input, so audit logs can be piped
    (``cat log.jsonl | apg audit stats -``). The same line parser is reused, so
    a malformed line still raises :class:`AuditFormatError` annotated with its
    line number. Unlike :func:`read_audit`, ``sys.stdin`` is *not* closed when
    iteration finishes -- the caller owns that stream's lifetime, and a missing
    file (``FileNotFoundError``) is impossible because nothing is opened.
    """
    return _iter_audit(sys.stdin, close=False)


def _iter_audit(fp: IO[str], *, close: bool = True) -> Iterator[AuditRecord]:
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
        if close:
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


# --- verdict filter (R31) -----------------------------------------------------


def filter_by_verdict(
    records: Iterable[AuditRecord],
    verdicts: Iterable[Verdict | str] | None,
) -> list[AuditRecord]:
    """Return only the records whose decision verdict is in ``verdicts``.

    ``verdicts`` may mix :class:`Verdict` members and their string values
    (e.g. ``"deny"``); a falsy value (``None`` or an empty collection) means
    "no filter" and returns every record unchanged. The result preserves input
    order and is materialized into a list so callers can summarize it more than
    once.

    Pure (no I/O), mirroring :func:`summarize_audit` / :func:`audit_stats_dict`,
    so the ``apg audit stats --verdict`` subcommand can apply it before handing
    the subset to either renderer. A filter that matches nothing yields an empty
    list, which both renderers treat as an empty log.
    """
    if not verdicts:
        return list(records)
    wanted = {v.value if isinstance(v, Verdict) else str(v) for v in verdicts}
    return [r for r in records if r.decision.verdict.value in wanted]


def filter_by_time(
    records: Iterable[AuditRecord],
    *,
    since: str | None = None,
    until: str | None = None,
) -> list[AuditRecord]:
    """Return only the records whose ``ts`` falls within ``[since, until]``.

    Both bounds are *inclusive* and compared **lexicographically** against
    ``record.ts``. Audit timestamps are ISO-8601 UTC
    (``YYYY-MM-DDTHH:MM:SS.ffffffZ``), so string ordering is chronological --
    the same property :func:`summarize_audit` already relies on for the span
    min/max -- and any ISO prefix works as a bound (e.g. ``2026-06-13`` selects
    from the start of that day; a full timestamp pins an exact instant). ``since``
    keeps records with ``ts >= since``; ``until`` keeps records with
    ``ts <= until``. A ``None`` bound is open on that side, and
    ``since=None, until=None`` returns every record unchanged.

    Pure (no I/O), mirroring :func:`filter_by_verdict` /
    :func:`summarize_audit`, so the ``apg audit stats --since/--until``
    subcommand can apply it before handing the subset to either renderer. The
    result preserves input order and is materialized into a list so callers can
    summarize it more than once. A window that matches nothing yields an empty
    list, which both renderers treat as an empty log.
    """
    if since is None and until is None:
        return list(records)
    out: list[AuditRecord] = []
    for r in records:
        if since is not None and r.ts < since:
            continue
        if until is not None and r.ts > until:
            continue
        out.append(r)
    return out


# --- audit stats summary (R29) ------------------------------------------------


def _pct(count: int, total: int) -> str:
    """Format ``count/total`` as a one-decimal percentage string (no ``%``)."""
    if total <= 0:
        return "0.0"
    return f"{100.0 * count / total:.1f}"


def _top(counter: Counter[str], n: int) -> list[tuple[str, int]]:
    """Most-frequent ``(name, count)`` pairs, ties broken by name ascending."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


#: Label used for decisions that carried no ``rule_id`` (the gateway's default
#: disposition rather than a named rule).
_NO_RULE = "(default - no rule)"


#: Label used for tool calls that carried no ``agent_id`` (unattributed
#: traffic rather than a named agent identity).
_NO_AGENT = "(unattributed - no agent_id)"


def summarize_audit(
    records: Iterable[AuditRecord],
    *,
    source: str | None = None,
    top_n: int = 5,
) -> list[str]:
    """Render a one-screen plain-text summary of an audit log as lines.

    The layout is deliberately stable (and test-covered): a header, the total
    record count, the first/last timestamp span, a fixed three-line verdict
    breakdown (always ``allow``/``deny``/``review`` in that order, even when a
    verdict has zero hits), the combined deny+review share, and the top
    ``top_n`` rules, tools, and agents by hit count. An empty log produces the
    header,
    a zero count, and a single explanatory line.

    Logic only: this function performs no I/O, mirroring ``cli._explain`` /
    ``cli._lint`` so callers (the ``apg audit stats`` subcommand and tests)
    can drive it directly.
    """
    recs = list(records)
    total = len(recs)
    lines: list[str] = []
    header = "audit log summary"
    if source is not None:
        header += f": {source}"
    lines.append(header)
    lines.append(f"records:     {total}")
    if total == 0:
        lines.append("(log is empty - no records to summarize)")
        return lines

    timestamps = [r.ts for r in recs]
    lines.append(f"span:        {min(timestamps)}  ..  {max(timestamps)}")

    verdict_counts: Counter[Verdict] = Counter(r.decision.verdict for r in recs)
    lines.append("verdicts:")
    for verdict in Verdict:
        count = verdict_counts.get(verdict, 0)
        lines.append(f"  {verdict.value:<7s}{count:>5d}  ({_pct(count, total)}%)")
    flagged = verdict_counts.get(Verdict.DENY, 0) + verdict_counts.get(
        Verdict.REVIEW, 0
    )
    lines.append(f"deny+review: {flagged}/{total}  ({_pct(flagged, total)}%)")

    rule_counts: Counter[str] = Counter(
        r.decision.rule_id if r.decision.rule_id else _NO_RULE for r in recs
    )
    lines.append(f"top rules (by hits, max {top_n}):")
    for name, count in _top(rule_counts, top_n):
        lines.append(f"  {count:>5d}  {name}")

    tool_counts: Counter[str] = Counter(r.call.tool_name for r in recs)
    lines.append(f"top tools (by hits, max {top_n}):")
    for name, count in _top(tool_counts, top_n):
        lines.append(f"  {count:>5d}  {name}")

    agent_counts: Counter[str] = Counter(
        r.call.agent_id if r.call.agent_id else _NO_AGENT for r in recs
    )
    lines.append(f"top agents (by hits, max {top_n}):")
    for name, count in _top(agent_counts, top_n):
        lines.append(f"  {count:>5d}  {name}")
    return lines


def audit_stats_dict(
    records: Iterable[AuditRecord],
    *,
    source: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Return the audit-log statistics as a JSON-serializable dict.

    This is the structured counterpart to :func:`summarize_audit`: it computes
    the same figures (record count, timestamp span, per-verdict counts and
    percentages for all four verdicts in enum order, the combined deny+review
    share, and the top ``top_n`` rules, tools, and agents by hit count) but
    returns them
    as a dict instead of rendering plain-text lines. Percentages are floats
    rounded to one decimal place, matching the text summary.

    Like :func:`summarize_audit`, this performs no I/O so callers (the
    ``apg audit stats --json`` subcommand and tests) can drive it directly. An
    empty log yields just ``{"source": ..., "records": 0}`` (``source`` omitted
    when ``None``), paralleling the text summary's empty-log shortcut.
    """
    recs = list(records)
    total = len(recs)
    result: dict[str, Any] = {}
    if source is not None:
        result["source"] = source
    result["records"] = total
    if total == 0:
        return result

    timestamps = [r.ts for r in recs]
    result["span"] = {"first": min(timestamps), "last": max(timestamps)}

    verdict_counts: Counter[Verdict] = Counter(r.decision.verdict for r in recs)
    result["verdicts"] = {
        verdict.value: {
            "count": verdict_counts.get(verdict, 0),
            "pct": float(_pct(verdict_counts.get(verdict, 0), total)),
        }
        for verdict in Verdict
    }
    flagged = verdict_counts.get(Verdict.DENY, 0) + verdict_counts.get(
        Verdict.REVIEW, 0
    )
    result["deny_review"] = {"count": flagged, "pct": float(_pct(flagged, total))}

    rule_counts: Counter[str] = Counter(
        r.decision.rule_id if r.decision.rule_id else _NO_RULE for r in recs
    )
    result["top_rules"] = [
        {"name": name, "count": count} for name, count in _top(rule_counts, top_n)
    ]

    tool_counts: Counter[str] = Counter(r.call.tool_name for r in recs)
    result["top_tools"] = [
        {"name": name, "count": count} for name, count in _top(tool_counts, top_n)
    ]

    agent_counts: Counter[str] = Counter(
        r.call.agent_id if r.call.agent_id else _NO_AGENT for r in recs
    )
    result["top_agents"] = [
        {"name": name, "count": count} for name, count in _top(agent_counts, top_n)
    ]
    return result


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
