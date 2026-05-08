"""Lightweight, standard-library-only benchmark harness for the gateway.

This module is intentionally dependency-free at runtime and produces
results suitable for both human inspection (a fixed-width text table)
and machine consumption (a JSON document). It is not pytest-benchmark
nor a competition-grade microbench framework; the goal is *reproducible,
comparable* numbers across releases — overhead per call and throughput
on the hot paths the gateway is responsible for:

* ``raw_call`` — the underlying tool function called directly, with no
  gateway, no audit. Establishes a baseline so the gateway-induced
  overhead is observable as a *delta* rather than an absolute number.
* ``gateway_allow`` — :class:`Gateway.execute` plus :meth:`Gateway.wrap_tool`
  against a policy whose first matching rule is an ``allow``. Exercises
  the decide-then-call hot path that production callers see most often.
* ``gateway_deny`` — same wrapper, against a policy whose first matching
  rule is a ``deny``. The wrapped callable raises
  :class:`PolicyDenied` *before* the underlying tool runs; the timed
  callable swallows the exception so percentiles reflect the refusal
  cost, not whatever the test harness does after the throw.
* ``gateway_allow_audit`` — gateway-allow with an in-process no-op
  :class:`AuditWriter` attached, isolating the audit-call dispatch cost
  from the JSON-encode + fsync cost of :class:`JsonlAuditWriter`.

The ``benchmark`` function is small enough to be useful as a building
block for ad-hoc one-offs:

    >>> from agent_policy_gateway.bench import benchmark
    >>> r = benchmark("noop", lambda: None, iterations=1000, warmup=100)
    >>> r.iterations, r.warmup
    (1000, 100)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agent_policy_gateway.core import Decision, ToolCall
from agent_policy_gateway.gateway import Gateway, PolicyDenied
from agent_policy_gateway.policy import load_policy_str

__all__ = [
    "BenchResult",
    "benchmark",
    "bench_main",
    "format_results_table",
    "make_scenario",
    "results_to_json",
    "run_default_suite",
    "scenario_gateway_allow",
    "scenario_gateway_allow_audit",
    "scenario_gateway_deny",
    "scenario_raw_call",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchResult:
    """Outcome of one benchmark scenario.

    All time fields are nanoseconds. ``iterations`` is the number of
    timed (post-warmup) calls; ``warmup`` is the number of un-timed
    calls executed before measurement to amortize first-call overheads
    (JIT warm-up, pyc loading, etc.).
    """

    name: str
    iterations: int
    warmup: int
    total_ns: int
    mean_ns: float
    min_ns: int
    max_ns: int
    p50_ns: int
    p95_ns: int
    p99_ns: int
    ops_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "total_ns": self.total_ns,
            "mean_ns": self.mean_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "p99_ns": self.p99_ns,
            "ops_per_sec": self.ops_per_sec,
        }


# ---------------------------------------------------------------------------
# Core timer
# ---------------------------------------------------------------------------


def _percentile_ns(sorted_samples: list[int], pct: float) -> int:
    """Return the ``pct``-th percentile of ``sorted_samples`` (0–100).

    Uses nearest-rank with ceil so a 1-sample input always returns the
    sole sample regardless of percentile. Inputs must already be sorted
    ascending.
    """
    if not sorted_samples:
        raise ValueError("cannot take percentile of empty samples")
    if pct < 0 or pct > 100:
        raise ValueError(f"percentile must be in [0, 100], got {pct}")
    n = len(sorted_samples)
    # nearest-rank, ceil-rounded; clamp to last index.
    rank = max(1, int(-(-pct * n // 100)))
    return sorted_samples[min(rank - 1, n - 1)]


def benchmark(
    name: str,
    fn: Callable[[], Any],
    *,
    iterations: int,
    warmup: int = 0,
) -> BenchResult:
    """Time ``fn`` ``iterations`` times after ``warmup`` un-timed calls.

    ``fn`` is invoked with no arguments. Per-call latency is recorded
    via :func:`time.perf_counter_ns`. The function returns a
    :class:`BenchResult` with mean, min, max, and the 50/95/99
    percentiles, plus a derived ``ops_per_sec`` from the total elapsed
    time across the timed window.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    for _ in range(warmup):
        fn()

    samples: list[int] = [0] * iterations
    perf_ns = time.perf_counter_ns
    # Tight loop: minimise overhead between the two clock reads.
    for i in range(iterations):
        t0 = perf_ns()
        fn()
        samples[i] = perf_ns() - t0

    total = sum(samples)
    samples_sorted = sorted(samples)
    return BenchResult(
        name=name,
        iterations=iterations,
        warmup=warmup,
        total_ns=total,
        mean_ns=total / iterations,
        min_ns=samples_sorted[0],
        max_ns=samples_sorted[-1],
        p50_ns=_percentile_ns(samples_sorted, 50),
        p95_ns=_percentile_ns(samples_sorted, 95),
        p99_ns=_percentile_ns(samples_sorted, 99),
        ops_per_sec=(iterations * 1_000_000_000) / total if total > 0 else float("inf"),
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


_ALLOW_POLICY_YAML = """
version: 1
name: bench-allow
rules:
  - id: allow-all
    when:
      tool: "*"
    effect:
      action: allow
"""

_DENY_POLICY_YAML = """
version: 1
name: bench-deny
rules:
  - id: deny-all
    when:
      tool: "*"
    effect:
      action: deny
"""


def _bench_tool(x: int = 1, y: int = 2) -> int:
    """A trivial pure-Python tool used by every scenario.

    Kept small so the benchmark measures the *gateway*, not the tool.
    """
    return x + y


def scenario_raw_call() -> Callable[[], Any]:
    """Baseline: call the tool function directly, no gateway."""
    fn = _bench_tool
    return lambda: fn(1, 2)


def scenario_gateway_allow() -> Callable[[], Any]:
    """Gateway.execute → wrap_tool with an allow policy, no audit."""
    gw = Gateway(policies=[load_policy_str(_ALLOW_POLICY_YAML)])
    wrapped = gw.wrap_tool(_bench_tool, tool_name="bench_tool")
    return lambda: wrapped(1, 2)


def scenario_gateway_deny() -> Callable[[], Any]:
    """Gateway with a deny policy; the wrapped callable raises PolicyDenied.

    The timed callable swallows :class:`PolicyDenied` so the percentile
    measures the refusal cost itself, not exception unwinding done by
    pytest or the harness.
    """
    gw = Gateway(policies=[load_policy_str(_DENY_POLICY_YAML)])
    wrapped = gw.wrap_tool(_bench_tool, tool_name="bench_tool")

    def call() -> None:
        try:
            wrapped(1, 2)
        except PolicyDenied:
            return None

    return call


class _NoopAuditWriter:
    """Counts calls without paying any I/O cost."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, call: ToolCall, decision: Decision) -> None:
        self.calls += 1


def scenario_gateway_allow_audit() -> Callable[[], Any]:
    """Gateway.execute + wrap_tool + an in-process no-op audit writer."""
    audit = _NoopAuditWriter()
    gw = Gateway(policies=[load_policy_str(_ALLOW_POLICY_YAML)], audit_writer=audit)
    wrapped = gw.wrap_tool(_bench_tool, tool_name="bench_tool")
    return lambda: wrapped(1, 2)


def make_scenario(name: str) -> Callable[[], Any]:
    """Look up a scenario factory by name. Raises ``KeyError`` on miss."""
    table = {
        "raw_call": scenario_raw_call,
        "gateway_allow": scenario_gateway_allow,
        "gateway_deny": scenario_gateway_deny,
        "gateway_allow_audit": scenario_gateway_allow_audit,
    }
    if name not in table:
        raise KeyError(name)
    return table[name]()


DEFAULT_SCENARIOS: tuple[str, ...] = (
    "raw_call",
    "gateway_allow",
    "gateway_deny",
    "gateway_allow_audit",
)


# ---------------------------------------------------------------------------
# Suite + reporting
# ---------------------------------------------------------------------------


def run_default_suite(
    *,
    iterations: int = 5_000,
    warmup: int = 500,
    scenarios: Iterable[str] | None = None,
) -> list[BenchResult]:
    """Run the default suite and return one :class:`BenchResult` per scenario.

    The order of returned results matches the order of ``scenarios``
    (which defaults to :data:`DEFAULT_SCENARIOS`), so callers can pair a
    scenario name with its result by index.
    """
    chosen = tuple(scenarios) if scenarios is not None else DEFAULT_SCENARIOS
    results: list[BenchResult] = []
    for name in chosen:
        fn = make_scenario(name)
        results.append(benchmark(name, fn, iterations=iterations, warmup=warmup))
    return results


def format_results_table(results: list[BenchResult]) -> str:
    """Render a fixed-width text table summarizing ``results``.

    The first column is the scenario name; remaining columns are the
    iteration count, mean / p50 / p95 / p99 in microseconds, and the
    derived throughput in ops/sec. Microseconds are chosen because the
    gateway hot path is comfortably above 1 µs and below 1 ms; percentile
    columns are right-aligned so visual diffs across runs are easy.
    """
    headers = ("scenario", "iters", "mean_us", "p50_us", "p95_us", "p99_us", "ops/sec")
    rows: list[tuple[str, ...]] = [headers]
    for r in results:
        rows.append(
            (
                r.name,
                str(r.iterations),
                f"{r.mean_ns / 1000:.2f}",
                f"{r.p50_ns / 1000:.2f}",
                f"{r.p95_ns / 1000:.2f}",
                f"{r.p99_ns / 1000:.2f}",
                f"{r.ops_per_sec:,.0f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    sep = "  "

    def fmt_row(row: tuple[str, ...]) -> str:
        return sep.join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        )

    lines = [fmt_row(rows[0])]
    lines.append(sep.join("-" * w for w in widths))
    for row in rows[1:]:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def results_to_json(results: list[BenchResult]) -> str:
    """Return a stable JSON document describing ``results``."""
    return json.dumps(
        {"results": [r.to_dict() for r in results]},
        indent=2,
        sort_keys=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class _CliConfig:
    iterations: int = 5_000
    warmup: int = 500
    scenarios: list[str] = field(default_factory=lambda: list(DEFAULT_SCENARIOS))
    as_json: bool = False


def _parse_args(argv: list[str] | None) -> _CliConfig:
    parser = argparse.ArgumentParser(
        prog="apg-bench",
        description=(
            "Run the agent-policy-gateway benchmark suite. "
            "Reports per-call overhead and throughput for the gateway's hot paths."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5_000,
        help="Timed calls per scenario (default: 5000).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="Un-timed warmup calls before measurement (default: 500).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=DEFAULT_SCENARIOS,
        help=(
            "Run only the named scenario; pass multiple times for a subset. "
            "If omitted, all scenarios run."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit results as a JSON document on stdout instead of a table.",
    )
    ns = parser.parse_args(argv)
    if ns.iterations <= 0:
        parser.error("--iterations must be > 0")
    if ns.warmup < 0:
        parser.error("--warmup must be >= 0")
    return _CliConfig(
        iterations=ns.iterations,
        warmup=ns.warmup,
        scenarios=list(ns.scenario) if ns.scenario else list(DEFAULT_SCENARIOS),
        as_json=bool(ns.json),
    )


def bench_main(argv: list[str] | None = None) -> int:
    """Console-script entry point. Returns the process exit code."""
    cfg = _parse_args(argv)
    results = run_default_suite(
        iterations=cfg.iterations,
        warmup=cfg.warmup,
        scenarios=cfg.scenarios,
    )
    out = results_to_json(results) if cfg.as_json else format_results_table(results)
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(bench_main())
