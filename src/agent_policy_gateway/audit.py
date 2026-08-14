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
import fnmatch
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
    "audit_allow_share",
    "audit_diff_dict",
    "audit_flagged_share",
    "audit_stats_csv",
    "audit_stats_dict",
    "audit_stats_section_csv",
    "exclude_by",
    "exclude_by_agent",
    "exclude_by_rule",
    "exclude_by_tool",
    "filter_by_agent",
    "filter_by_rule",
    "filter_by_time",
    "filter_by_tool",
    "filter_by_verdict",
    "format_record",
    "read_audit",
    "read_audit_stdin",
    "replay_main",
    "summarize_audit",
    "summarize_audit_diff",
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


# --- tool-name filter (R35) ---------------------------------------------------


def filter_by_tool(
    records: Iterable[AuditRecord],
    patterns: Iterable[str] | None,
) -> list[AuditRecord]:
    """Return only the records whose ``call.tool_name`` matches ``patterns``.

    Each pattern is an :func:`fnmatch.fnmatchcase` glob (``*``, ``?``, ``[seq]``);
    a record is kept when its tool name matches *any* pattern (union). A literal
    pattern with no wildcards therefore selects an exact tool name. Matching is
    case-sensitive (``fnmatchcase``), mirroring the exactness of
    :func:`filter_by_verdict`. A falsy ``patterns`` value (``None`` or an empty
    collection) means "no filter" and returns every record unchanged.

    Pure (no I/O), mirroring :func:`filter_by_verdict` / :func:`filter_by_time`,
    so the ``apg audit stats --tool`` subcommand can apply it before handing the
    subset to either renderer. The result preserves input order and is
    materialized into a list so callers can summarize it more than once. A set
    of patterns that matches nothing yields an empty list, which both renderers
    treat as an empty log.
    """
    if not patterns:
        return list(records)
    pats = list(patterns)
    return [
        r
        for r in records
        if any(fnmatch.fnmatchcase(r.call.tool_name, p) for p in pats)
    ]


def filter_by_agent(
    records: Iterable[AuditRecord],
    patterns: Iterable[str] | None,
) -> list[AuditRecord]:
    """Return only the records whose ``call.agent_id`` matches ``patterns``.

    Mirrors :func:`filter_by_tool` (R35): each pattern is an
    :func:`fnmatch.fnmatchcase` glob (``*``, ``?``, ``[seq]``) and a record is
    kept when its agent id matches *any* pattern (union); a literal pattern
    with no wildcards selects an exact id. Matching is case-sensitive. A falsy
    ``patterns`` value (``None`` or an empty collection) means "no filter" and
    returns every record unchanged.

    Records that carry no ``agent_id`` (the unattributed bucket) are matched
    under the :data:`_NO_AGENT` sentinel label, reusing the R33 semantics from
    the agent breakdown. Callers therefore select unattributed traffic with an
    explicit ``filter_by_agent(records, [_NO_AGENT])`` (the CLI surfaces the
    literal sentinel string in ``--help``).

    Pure (no I/O), order-preserving, and materialized into a list so callers
    can summarize the subset more than once. A pattern set that matches
    nothing yields an empty list, which both renderers treat as an empty log.
    """
    if not patterns:
        return list(records)
    pats = list(patterns)
    return [
        r
        for r in records
        if any(
            fnmatch.fnmatchcase(r.call.agent_id or _NO_AGENT, p) for p in pats
        )
    ]


def filter_by_rule(
    records: Iterable[AuditRecord],
    patterns: Iterable[str] | None,
) -> list[AuditRecord]:
    """Return only the records whose ``decision.rule_id`` matches ``patterns``.

    Mirrors :func:`filter_by_agent` (R36) but matches the *matched rule id*
    rather than the agent: each pattern is an :func:`fnmatch.fnmatchcase` glob
    (``*``, ``?``, ``[seq]``) and a record is kept when its rule id matches
    *any* pattern (union); a literal pattern with no wildcards selects an exact
    id. Matching is case-sensitive. A falsy ``patterns`` value (``None`` or an
    empty collection) means "no filter" and returns every record unchanged.

    Decisions that carried no ``rule_id`` (the gateway's default, no-rule
    bucket) are matched under the :data:`_NO_RULE` sentinel label, reusing the
    R33 semantics from the top-rules breakdown. Callers therefore select
    default/unruled traffic with an explicit
    ``filter_by_rule(records, [_NO_RULE])`` (the CLI surfaces the literal
    sentinel string in ``--help``).

    Pure (no I/O), order-preserving, and materialized into a list so callers
    can summarize the subset more than once. A pattern set that matches
    nothing yields an empty list, which both renderers treat as an empty log.
    """
    if not patterns:
        return list(records)
    pats = list(patterns)
    return [
        r
        for r in records
        if any(
            fnmatch.fnmatchcase(r.decision.rule_id or _NO_RULE, p) for p in pats
        )
    ]


# --- negative filters (R43) ---------------------------------------------------


#: Value extractors for :func:`exclude_by`, keyed by filter axis. Each one
#: returns the *same* string the matching include filter globs against, so the
#: sentinel buckets (``_NO_AGENT`` / ``_NO_RULE``) are excludable by their
#: literal labels exactly as they are selectable.
_EXCLUDE_KEYS: dict[str, Callable[[AuditRecord], str]] = {
    "tool": lambda r: r.call.tool_name,
    "agent": lambda r: r.call.agent_id or _NO_AGENT,
    "rule": lambda r: r.decision.rule_id or _NO_RULE,
}


def exclude_by(
    records: Iterable[AuditRecord],
    key: str,
    patterns: Iterable[str] | None,
) -> list[AuditRecord]:
    """Return the records that do **not** match ``patterns`` on axis ``key``.

    The inverse of :func:`filter_by_tool` / :func:`filter_by_agent` /
    :func:`filter_by_rule`, sharing their matching rules: ``key`` selects the
    axis (``"tool"`` -> ``call.tool_name``, ``"agent"`` -> ``call.agent_id``,
    ``"rule"`` -> ``decision.rule_id``), each pattern is an
    :func:`fnmatch.fnmatchcase` glob (``*``, ``?``, ``[seq]``), matching is
    case-sensitive, and the unattributed/default buckets are addressed through
    the :data:`_NO_AGENT` / :data:`_NO_RULE` sentinel labels. A record is
    *dropped* when it matches **any** pattern (union), so exclusions never
    interact: adding a pattern can only shrink the result.

    Callers apply the include filters first and the exclusions second, which
    makes ``--tool 'send_*' --exclude-tool 'send_test'`` mean "the send family
    minus the test tool". A falsy ``patterns`` value (``None`` or an empty
    collection) means "exclude nothing" and returns every record unchanged.

    Pure (no I/O), order-preserving, and materialized into a list so callers
    can summarize the subset more than once. Excluding everything yields an
    empty list, which both renderers treat as an empty log. An unknown ``key``
    raises :class:`ValueError`.
    """
    try:
        value_of = _EXCLUDE_KEYS[key]
    except KeyError:
        valid = ", ".join(sorted(_EXCLUDE_KEYS))
        raise ValueError(
            f"unknown exclude key {key!r} (expected one of: {valid})"
        ) from None
    if not patterns:
        return list(records)
    pats = list(patterns)
    return [
        r
        for r in records
        if not any(fnmatch.fnmatchcase(value_of(r), p) for p in pats)
    ]


def exclude_by_tool(
    records: Iterable[AuditRecord],
    patterns: Iterable[str] | None,
) -> list[AuditRecord]:
    """Drop the records whose ``call.tool_name`` matches any of ``patterns``."""
    return exclude_by(records, "tool", patterns)


def exclude_by_agent(
    records: Iterable[AuditRecord],
    patterns: Iterable[str] | None,
) -> list[AuditRecord]:
    """Drop the records whose ``call.agent_id`` matches any of ``patterns``."""
    return exclude_by(records, "agent", patterns)


def exclude_by_rule(
    records: Iterable[AuditRecord],
    patterns: Iterable[str] | None,
) -> list[AuditRecord]:
    """Drop the records whose ``decision.rule_id`` matches any of ``patterns``."""
    return exclude_by(records, "rule", patterns)


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


def audit_flagged_share(records: Iterable[AuditRecord]) -> float:
    """Return the combined deny+review ("flagged") share as a percentage.

    This is the same figure rendered as ``deny+review`` by
    :func:`summarize_audit` and stored under ``deny_review.pct`` by
    :func:`audit_stats_dict`, exposed on its own so the ``apg audit stats
    --fail-over`` CI gate (R39) can threshold it without re-rendering. The
    value is an exact (unrounded) percentage in ``[0.0, 100.0]``; an empty log
    yields ``0.0`` so it is never "over" any non-negative threshold.

    Pure (no I/O), mirroring the other ``audit stats`` helpers, so the
    subcommand and tests can drive it directly.
    """
    recs = list(records)
    total = len(recs)
    if total == 0:
        return 0.0
    flagged = sum(
        1
        for r in recs
        if r.decision.verdict in (Verdict.DENY, Verdict.REVIEW)
    )
    return 100.0 * flagged / total


def audit_allow_share(records: Iterable[AuditRecord]) -> float:
    """Return the ``allow`` share as a percentage.

    The mirror of :func:`audit_flagged_share`: this is the same figure rendered
    as the ``allow`` line by :func:`summarize_audit` and stored under
    ``verdicts.allow.pct`` by :func:`audit_stats_dict`, exposed on its own so
    the ``apg audit stats --fail-under`` CI gate (R45) can threshold it without
    re-rendering. The value is an exact (unrounded) percentage in
    ``[0.0, 100.0]``; an empty log yields ``0.0``.

    Pure (no I/O), mirroring the other ``audit stats`` helpers, so the
    subcommand and tests can drive it directly.
    """
    recs = list(records)
    total = len(recs)
    if total == 0:
        return 0.0
    allowed = sum(1 for r in recs if r.decision.verdict is Verdict.ALLOW)
    return 100.0 * allowed / total


def summarize_audit(
    records: Iterable[AuditRecord],
    *,
    source: str | None = None,
    top_n: int = 5,
    top_rules: int | None = None,
    top_tools: int | None = None,
    top_agents: int | None = None,
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
    # R38: each list may cap independently; an omitted per-list cap falls back
    # to the shared ``top_n`` so omitting all three is byte-for-byte unchanged.
    rules_n = top_n if top_rules is None else top_rules
    tools_n = top_n if top_tools is None else top_tools
    agents_n = top_n if top_agents is None else top_agents
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
    lines.append(f"top rules (by hits, max {rules_n}):")
    for name, count in _top(rule_counts, rules_n):
        lines.append(f"  {count:>5d}  {name}")

    tool_counts: Counter[str] = Counter(r.call.tool_name for r in recs)
    lines.append(f"top tools (by hits, max {tools_n}):")
    for name, count in _top(tool_counts, tools_n):
        lines.append(f"  {count:>5d}  {name}")

    agent_counts: Counter[str] = Counter(
        r.call.agent_id if r.call.agent_id else _NO_AGENT for r in recs
    )
    lines.append(f"top agents (by hits, max {agents_n}):")
    for name, count in _top(agent_counts, agents_n):
        lines.append(f"  {count:>5d}  {name}")
    return lines


def audit_stats_dict(
    records: Iterable[AuditRecord],
    *,
    source: str | None = None,
    top_n: int = 5,
    top_rules: int | None = None,
    top_tools: int | None = None,
    top_agents: int | None = None,
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
    # R38: per-list caps fall back to the shared ``top_n`` when omitted, so the
    # default output is unchanged.
    rules_n = top_n if top_rules is None else top_rules
    tools_n = top_n if top_tools is None else top_tools
    agents_n = top_n if top_agents is None else top_agents
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
        {"name": name, "count": count} for name, count in _top(rule_counts, rules_n)
    ]

    tool_counts: Counter[str] = Counter(r.call.tool_name for r in recs)
    result["top_tools"] = [
        {"name": name, "count": count} for name, count in _top(tool_counts, tools_n)
    ]

    agent_counts: Counter[str] = Counter(
        r.call.agent_id if r.call.agent_id else _NO_AGENT for r in recs
    )
    result["top_agents"] = [
        {"name": name, "count": count} for name, count in _top(agent_counts, agents_n)
    ]
    return result


def audit_stats_csv(
    records: Iterable[AuditRecord],
    *,
    source: str | None = None,
) -> list[str]:
    """Return the per-verdict counts and percentages as CSV rows.

    The CSV counterpart to :func:`summarize_audit` / :func:`audit_stats_dict`,
    intended for piping into a spreadsheet (``apg audit stats --csv``). The
    output is a ``verdict,count,pct`` header followed by one row per verdict in
    :class:`Verdict` enum order (always all four, even when a verdict has zero
    hits) and a trailing ``deny+review`` row combining the flagged verdicts.
    Counts and percentages match the text and JSON renderers: percentages are
    one-decimal strings produced by the shared :func:`_pct` helper.

    Like :func:`summarize_audit` / :func:`audit_stats_dict`, this performs no
    I/O so callers (the ``apg audit stats --csv`` subcommand and tests) can
    drive it directly. ``source`` is accepted for signature parity with the
    sibling renderers but does not appear in the CSV body. An empty log yields
    just the header row, paralleling the empty-log shortcut of the text and
    JSON renderers.

    The emitted fields never contain commas or quotes (verdict names are bare
    identifiers, the combined row is the literal ``deny+review``), so the rows
    are valid CSV without any quoting.
    """
    del source  # parity with sibling renderers; not part of the CSV body
    recs = list(records)
    total = len(recs)
    header = "verdict,count,pct"
    if total == 0:
        return [header]

    verdict_counts: Counter[Verdict] = Counter(r.decision.verdict for r in recs)
    lines = [header]
    for verdict in Verdict:
        count = verdict_counts.get(verdict, 0)
        lines.append(f"{verdict.value},{count},{_pct(count, total)}")
    flagged = verdict_counts.get(Verdict.DENY, 0) + verdict_counts.get(
        Verdict.REVIEW, 0
    )
    lines.append(f"deny+review,{flagged},{_pct(flagged, total)}")
    return lines


# --- audit stats section CSV (R42) --------------------------------------------


#: The breakdowns ``audit_stats_section_csv`` can emit, mapped to the singular
#: noun used as the first CSV header field. ``verdicts`` is the R40 default and
#: is delegated verbatim to :func:`audit_stats_csv`.
CSV_SECTIONS: tuple[str, ...] = ("verdicts", "rules", "tools", "agents")

_SECTION_LABEL = {"rules": "rule", "tools": "tool", "agents": "agent"}


def _csv_field(value: str) -> str:
    """Quote a CSV field only when it needs it (comma, quote, or newline)."""
    if any(c in value for c in ',"\r\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def audit_stats_section_csv(
    records: Iterable[AuditRecord],
    section: str = "verdicts",
    *,
    source: str | None = None,
    top_n: int = 5,
    top_rules: int | None = None,
    top_tools: int | None = None,
    top_agents: int | None = None,
) -> list[str]:
    """Return one breakdown of the audit statistics as CSV rows.

    Generalizes :func:`audit_stats_csv` beyond the per-verdict table
    (``apg audit stats --csv --csv-section ...``). ``section`` selects the
    breakdown:

    * ``"verdicts"`` (default) delegates to :func:`audit_stats_csv`, so the
      R40 output is reproduced byte-for-byte and ``--csv`` alone is unchanged.
    * ``"rules"`` / ``"tools"`` / ``"agents"`` emit the corresponding ranked
      top-N list as a ``rule,count,pct`` / ``tool,count,pct`` /
      ``agent,count,pct`` header followed by one row per entry, in the same
      order (count descending, ties by name ascending) that
      :func:`summarize_audit` and :func:`audit_stats_dict` use. The per-list
      caps behave as elsewhere: an omitted ``top_rules``/``top_tools``/
      ``top_agents`` falls back to the shared ``top_n``.

    Percentages are the entry's share of the (already filtered) record total,
    formatted by the shared :func:`_pct` helper, matching the sibling
    renderers. The unnamed buckets use the same ``_NO_RULE`` / ``_NO_AGENT``
    sentinel labels as the text and JSON renderers.

    Like the other ``audit stats`` helpers this performs no I/O, so the
    subcommand and tests can drive it directly. An empty log yields just the
    header row. Names are CSV-quoted only when they contain a comma, quote, or
    newline, so ordinary tool/rule/agent identifiers stay bare.

    Raises :class:`ValueError` for an unknown ``section``.
    """
    if section not in CSV_SECTIONS:
        raise ValueError(
            f"unknown csv section: {section!r} "
            f"(expected one of {', '.join(CSV_SECTIONS)})"
        )
    if section == "verdicts":
        return audit_stats_csv(records, source=source)

    recs = list(records)
    total = len(recs)
    label = _SECTION_LABEL[section]
    header = f"{label},count,pct"
    if total == 0:
        return [header]

    if section == "rules":
        counts: Counter[str] = Counter(
            r.decision.rule_id if r.decision.rule_id else _NO_RULE for r in recs
        )
        cap = top_n if top_rules is None else top_rules
    elif section == "tools":
        counts = Counter(r.call.tool_name for r in recs)
        cap = top_n if top_tools is None else top_tools
    else:  # "agents"
        counts = Counter(
            r.call.agent_id if r.call.agent_id else _NO_AGENT for r in recs
        )
        cap = top_n if top_agents is None else top_agents

    lines = [header]
    for name, count in _top(counts, cap):
        lines.append(f"{_csv_field(name)},{count},{_pct(count, total)}")
    return lines



# --- audit log diff (R47) -----------------------------------------------------


#: Axes ``audit_diff_dict`` reports movement on, mapped to the extractor that
#: produces the label each record contributes. The sentinel buckets
#: (``_NO_RULE`` / ``_NO_AGENT``) are reused verbatim, so the unnamed traffic
#: is diffed under the same labels the stats renderers already print.
_DIFF_AXES: dict[str, Callable[[AuditRecord], str]] = {
    "rules": lambda r: r.decision.rule_id or _NO_RULE,
    "tools": lambda r: r.call.tool_name,
    "agents": lambda r: r.call.agent_id or _NO_AGENT,
}


def _delta_pct(old_pct: str, new_pct: str) -> float:
    """Signed one-decimal difference between two :func:`_pct` strings.

    The delta is computed from the *printed* (already one-decimal) shares
    rather than the exact ratios, so every rendered line is self-consistent:
    the reported delta is exactly ``new - old`` of the two numbers next to it.
    (The CI gates in :func:`audit_flagged_share` / :func:`audit_allow_share`
    deliberately keep full precision -- they threshold, they do not display.)
    Negative zero is normalized to ``0.0`` so a no-change line prints ``+0.0``.
    """
    delta = round(float(new_pct) - float(old_pct), 1)
    return 0.0 if delta == 0 else delta


def _ranks(counter: Counter[str]) -> dict[str, int]:
    """1-based rank per name, ordered like :func:`_top` (count desc, name asc)."""
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return {name: i for i, (name, _count) in enumerate(ordered, start=1)}


def _axis_movement(
    old_counts: Counter[str],
    new_counts: Counter[str],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Entries that appeared, disappeared, or moved most on one axis.

    Each entry carries both sides' hit counts and 1-based ranks, the signed
    count delta, and a ``status`` of ``"added"`` (absent from the old log),
    ``"removed"`` (absent from the new log), or ``"moved"``. ``rank_delta`` is
    ``old_rank - new_rank`` -- **positive means the entry climbed** the ranked
    list -- and is ``None`` whenever one side is missing, since a rank cannot
    be subtracted from nothing.

    Entries whose count *and* rank are identical on both sides are dropped, so
    two identical logs produce an empty list on every axis. The survivors are
    ordered by how much they moved: appearances/disappearances first, then the
    largest absolute rank change, then the largest absolute count change, ties
    broken by name ascending. The first ``top_n`` are returned (``top_n <= 0``
    yields an empty list).
    """
    old_ranks = _ranks(old_counts)
    new_ranks = _ranks(new_counts)
    entries: list[dict[str, Any]] = []
    for name in sorted(set(old_counts) | set(new_counts)):
        old_count = old_counts.get(name, 0)
        new_count = new_counts.get(name, 0)
        old_rank = old_ranks.get(name)
        new_rank = new_ranks.get(name)
        if old_count == new_count and old_rank == new_rank:
            continue
        if old_rank is None:
            status = "added"
        elif new_rank is None:
            status = "removed"
        else:
            status = "moved"
        rank_delta = (
            None
            if old_rank is None or new_rank is None
            else old_rank - new_rank
        )
        entries.append(
            {
                "name": name,
                "status": status,
                "old_count": old_count,
                "new_count": new_count,
                "delta": new_count - old_count,
                "old_rank": old_rank,
                "new_rank": new_rank,
                "rank_delta": rank_delta,
            }
        )
    entries.sort(
        key=lambda e: (
            0 if e["status"] == "moved" else -1,
            -abs(e["rank_delta"] or 0),
            -abs(e["delta"]),
            e["name"],
        )
    )
    return entries[: max(top_n, 0)]


def audit_diff_dict(
    old: Iterable[AuditRecord],
    new: Iterable[AuditRecord],
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Compare two audit logs and return the differences as a dict.

    The audit-log counterpart to ``apg policy diff``: where that command
    compares two policies by the decisions they *would* make, this compares two
    logs by the decisions that were actually recorded. The result is
    JSON-serializable and contains

    * ``records`` -- ``{"old", "new", "delta"}`` record counts,
    * ``verdicts`` -- one entry per :class:`Verdict` member, in enum order,
      each ``{"old": {"count", "pct"}, "new": {"count", "pct"},
      "delta": {"count", "pct"}}``,
    * ``deny_review`` -- the same shape for the combined flagged share,
    * ``rules`` / ``tools`` / ``agents`` -- up to ``top_n`` entries per axis
      describing what appeared, disappeared, or moved most in rank (see
      :func:`_axis_movement` for the entry shape and ordering).

    Percentages are one-decimal floats matching :func:`audit_stats_dict`, and
    each ``delta.pct`` is the difference of the two printed shares (see
    :func:`_delta_pct`). Both sides may be empty: an empty log contributes zero
    counts and ``0.0`` shares, so a first-ever log diffs cleanly against
    nothing.

    Pure (no I/O), like every other ``audit stats`` helper, so the
    ``apg audit diff`` subcommand and tests can drive it directly.
    """
    old_recs = list(old)
    new_recs = list(new)
    old_total = len(old_recs)
    new_total = len(new_recs)

    result: dict[str, Any] = {
        "records": {
            "old": old_total,
            "new": new_total,
            "delta": new_total - old_total,
        }
    }

    old_verdicts: Counter[Verdict] = Counter(r.decision.verdict for r in old_recs)
    new_verdicts: Counter[Verdict] = Counter(r.decision.verdict for r in new_recs)

    def _side(count: int, total: int) -> dict[str, Any]:
        return {"count": count, "pct": float(_pct(count, total))}

    def _block(old_count: int, new_count: int) -> dict[str, Any]:
        old_pct = _pct(old_count, old_total)
        new_pct = _pct(new_count, new_total)
        return {
            "old": _side(old_count, old_total),
            "new": _side(new_count, new_total),
            "delta": {
                "count": new_count - old_count,
                "pct": _delta_pct(old_pct, new_pct),
            },
        }

    result["verdicts"] = {
        verdict.value: _block(
            old_verdicts.get(verdict, 0), new_verdicts.get(verdict, 0)
        )
        for verdict in Verdict
    }
    old_flagged = old_verdicts.get(Verdict.DENY, 0) + old_verdicts.get(
        Verdict.REVIEW, 0
    )
    new_flagged = new_verdicts.get(Verdict.DENY, 0) + new_verdicts.get(
        Verdict.REVIEW, 0
    )
    result["deny_review"] = _block(old_flagged, new_flagged)

    for axis, value_of in _DIFF_AXES.items():
        result[axis] = _axis_movement(
            Counter(value_of(r) for r in old_recs),
            Counter(value_of(r) for r in new_recs),
            top_n=top_n,
        )
    return result


#: Marker printed in front of each movement entry, by status.
_DIFF_MARK = {"added": "+", "removed": "-", "moved": " "}


def _format_rank(rank: int | None) -> str:
    """Render a 1-based rank, or ``-`` for a side where the name is absent."""
    return "-" if rank is None else str(rank)


def _movement_lines(entries: list[dict[str, Any]], label: str, cap: int) -> list[str]:
    """Render one axis of :func:`audit_diff_dict` output as text lines."""
    lines = [f"top {label} changes (by movement, max {cap}):"]
    if not entries:
        lines.append("  (no change)")
        return lines
    for e in entries:
        mark = _DIFF_MARK[e["status"]]
        lines.append(
            f"  {mark} {e['name']}: {e['old_count']} -> {e['new_count']} "
            f"({e['delta']:+d})  rank "
            f"{_format_rank(e['old_rank'])} -> {_format_rank(e['new_rank'])}"
        )
    return lines


def summarize_audit_diff(
    old: Iterable[AuditRecord],
    new: Iterable[AuditRecord],
    *,
    old_source: str | None = None,
    new_source: str | None = None,
    top_n: int = 5,
) -> list[str]:
    """Render :func:`audit_diff_dict` as a stable plain-text block of lines.

    The text counterpart to :func:`audit_diff_dict`, mirroring
    :func:`summarize_audit`: a header naming both logs (omitted per side when
    the source is ``None``), the record-count delta, a fixed per-verdict
    breakdown in :class:`Verdict` enum order showing ``old -> new`` counts and
    shares with signed one-decimal deltas, the combined ``deny+review`` line,
    and up to ``top_n`` movement entries for rules, tools, and agents. Axes
    with nothing to report print ``(no change)`` rather than an empty block, so
    the layout is the same height whatever the input.

    Performs no I/O, so the ``apg audit diff`` subcommand and tests can drive
    it directly.
    """
    old_recs = list(old)
    new_recs = list(new)
    diff = audit_diff_dict(old_recs, new_recs, top_n=top_n)

    header = "audit log diff"
    if old_source is not None or new_source is not None:
        header += f": {old_source or '(none)'} -> {new_source or '(none)'}"
    lines = [header]
    rec = diff["records"]
    lines.append(
        f"records:     {rec['old']} -> {rec['new']}  ({rec['delta']:+d})"
    )
    if rec["old"] == 0 and rec["new"] == 0:
        lines.append("(both logs are empty - nothing to compare)")
        return lines

    lines.append("verdicts:")
    for verdict in Verdict:
        block = diff["verdicts"][verdict.value]
        lines.append(
            f"  {verdict.value:<7s}{block['old']['count']:>5d} -> "
            f"{block['new']['count']:<5d} ({block['delta']['count']:+d})  "
            f"{block['old']['pct']:.1f}% -> {block['new']['pct']:.1f}% "
            f"({block['delta']['pct']:+.1f})"
        )
    flagged = diff["deny_review"]
    lines.append(
        f"deny+review: {flagged['old']['count']}/{rec['old']} -> "
        f"{flagged['new']['count']}/{rec['new']}  "
        f"({flagged['delta']['count']:+d})  "
        f"{flagged['old']['pct']:.1f}% -> {flagged['new']['pct']:.1f}% "
        f"({flagged['delta']['pct']:+.1f})"
    )
    for axis in _DIFF_AXES:
        lines.extend(_movement_lines(diff[axis], _SECTION_LABEL[axis], top_n))
    return lines


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
