"""Gateway mediation overhead micro-benchmark (R60).

Every number the R55/R56/R59 benchmarks report is about *what the
gateway prevents*; this module measures *what the gateway costs*. Two
complementary measurements, both built on the R12 :mod:`bench` timer
(``perf_counter_ns``, un-timed warmup, nearest-rank percentiles):

**The mediation ladder** times one trivial pure-Python tool call through
:meth:`Gateway.execute`, adding one bookkeeping feature per rung, so the
per-feature cost is readable as the delta between adjacent rungs (and
the total mediation cost as the delta against the raw baseline):

* ``raw_call`` — the tool function called directly; the baseline.
* ``mediated_allow`` — a one-rule wildcard allow policy: the minimal
  decide-then-call path (policy evaluation only).
* ``mediated_taint`` — plus taint bookkeeping: a tainted input label,
  a :class:`ToolTaintSpec` with adds, and a taint-conditioned rule, so
  matching inspects the label and :func:`propagate` runs for real.
* ``mediated_history`` — plus history tracking (R53):
  ``track_history=True`` and a chain-conditioned rule. The gateway's
  history is reset after every timed call (public
  :meth:`Gateway.reset_history`) so the recorded length stays constant
  at zero; the reset's dict-clear cost is included in the rung, which
  slightly *overstates* the per-call cost — the conservative direction.
* ``mediated_full`` — taint + history + a no-op in-process audit
  writer: everything on.
* ``value_ledger_roundtrip`` — the R57 per-value bookkeeping measured
  on its own (it lives in the adapter wrapper, not
  :meth:`Gateway.execute`): one :meth:`ValueLedger.record` of an
  already-known value plus one :meth:`ValueLedger.labels_for_args` over
  a two-argument mapping, against a ledger pre-filled with
  :data:`LEDGER_PREFILL` seeded entries.

**The scaling sweeps** time the pure :meth:`Gateway.decide` path (no
tool execution, nothing recorded) while one workload parameter grows:

* ``rule_count`` — an *N*-rule policy where only the **last** rule
  matches the benchmarked tool, forcing the worst-case full scan.
* ``history_length`` — a session history of *H* recorded calls
  (pre-populated through the public execute path) walked by a chain
  rule whose ``no_prior`` matcher can never match, forcing the full
  O(H) walk plus the per-decide history-view copy.
* ``taint_set_size`` — an input label carrying *S* distinct integrity
  sources through a spec-add and a taint-conditioned rule, so the
  per-dimension set algebra runs at size *S*.

Workloads are synthetic and **seeded**: rule tool-names, history
sources, and taint sources all derive from ``random.Random(seed)``, so
two runs with the same seed measure byte-identical workloads and the
numbers in ``docs/benchmarks/overhead.md`` are regenerable by one
pinned command. Absolute numbers are machine-dependent; the *shape*
(deltas and growth curves) is the finding.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from agent_policy_gateway.bench import BenchResult, benchmark
from agent_policy_gateway.core import TaintLabel, ToolCall
from agent_policy_gateway.gateway import Gateway
from agent_policy_gateway.policy import (
    Action,
    ChainCondition,
    Effect,
    Policy,
    PriorCallMatcher,
    Rule,
    Selector,
    TaintCondition,
)
from agent_policy_gateway.taint import ToolTaintSpec
from agent_policy_gateway.value_flow import ValueLedger

__all__ = [
    "DEFAULT_SIZES",
    "LADDER_RUNGS",
    "LEDGER_PREFILL",
    "OverheadReport",
    "SWEEP_NAMES",
    "SweepPoint",
    "format_report",
    "make_history_workload",
    "make_rule_policy",
    "make_taint_workload",
    "overhead_main",
    "report_to_json",
    "run_ladder",
    "run_overhead_suite",
    "run_sweep",
]

BENCH_TOOL = "bench_tool"

LADDER_RUNGS: tuple[str, ...] = (
    "raw_call",
    "mediated_allow",
    "mediated_taint",
    "mediated_history",
    "mediated_full",
    "value_ledger_roundtrip",
)

SWEEP_NAMES: tuple[str, ...] = ("rule_count", "history_length", "taint_set_size")

DEFAULT_SIZES: tuple[int, ...] = (1, 16, 64, 256, 1024)

# Entries pre-recorded in the ledger rung, so the measured dict hit runs
# against a realistically non-empty table rather than a 1-entry one.
LEDGER_PREFILL = 1024

# The taint ladder rung uses a small fixed label; label *size* scaling is
# the taint_set_size sweep's job.
_LADDER_TAINT_SOURCES = 4


def _bench_tool(x: int = 1, y: int = 2) -> int:
    """The trivial pure-Python tool every rung mediates."""
    return x + y


def _allow_all_rule() -> Rule:
    return Rule(
        id="allow-rest",
        when=Selector(tool="*"),
        effect=Effect(action=Action.ALLOW),
    )


# ---------------------------------------------------------------------------
# Seeded synthetic workloads
# ---------------------------------------------------------------------------


def make_rule_policy(n_rules: int, *, seed: int = 0) -> Policy:
    """An ``n_rules``-rule policy where only the **last** rule matches
    :data:`BENCH_TOOL`.

    The first ``n_rules - 1`` rules select seeded synthetic tool names
    that can never equal :data:`BENCH_TOOL`, so a decision for the
    benchmarked tool scans the entire rule list — the worst case for
    first-match evaluation.
    """
    if n_rules < 1:
        raise ValueError(f"n_rules must be >= 1, got {n_rules}")
    rng = random.Random(seed)
    rules = [
        Rule(
            id=f"miss-{i:04d}",
            when=Selector(tool=f"other-{rng.getrandbits(32):08x}-{i}"),
            effect=Effect(action=Action.DENY),
        )
        for i in range(n_rules - 1)
    ]
    rules.append(
        Rule(
            id="match-last",
            when=Selector(tool=BENCH_TOOL),
            effect=Effect(action=Action.ALLOW),
        )
    )
    return Policy(name=f"overhead-rules-{n_rules}", rules=tuple(rules))


def make_history_workload(
    history_length: int, *, seed: int = 0
) -> tuple[Gateway, ToolCall]:
    """A history-tracking gateway with ``history_length`` recorded calls
    and a chain rule that walks all of them.

    The history is populated through the public execute path (one
    mediated setup call per entry, each with a seeded taint source on
    its output label), so the entries are exactly what production
    recording produces. The matching rule's ``no_prior`` matcher globs a
    tool name no entry carries: the clause can only be verified by
    scanning the full history, and always succeeds, so the measured
    decision is an ALLOW that paid the O(H) walk.
    """
    if history_length < 0:
        raise ValueError(f"history_length must be >= 0, got {history_length}")
    rng = random.Random(seed)
    policy = Policy(
        name=f"overhead-history-{history_length}",
        rules=(
            Rule(
                id="chain-scan",
                when=Selector(
                    tool=BENCH_TOOL,
                    chain=ChainCondition(
                        no_prior=(PriorCallMatcher(tool="never-matches-*"),)
                    ),
                ),
                effect=Effect(action=Action.ALLOW),
            ),
            _allow_all_rule(),
        ),
    )
    gw = Gateway(policies=[policy], track_history=True)
    for i in range(history_length):
        tool = f"setup-{i:04d}"
        gw.register_tool(
            tool, ToolTaintSpec.of(adds=(f"src-{rng.getrandbits(32):08x}",))
        )
        gw.execute(ToolCall(tool_name=tool, agent_id="bench-agent"), _bench_tool)
    call = ToolCall(tool_name=BENCH_TOOL, agent_id="bench-agent")
    return gw, call


def make_taint_workload(
    taint_set_size: int, *, seed: int = 0
) -> tuple[Gateway, ToolCall]:
    """A gateway whose decision joins and inspects a ``taint_set_size``-
    source input label.

    The call's input label carries ``taint_set_size`` distinct seeded
    integrity sources; the tool's spec adds one more, so
    :func:`propagate` really joins at size *S*; and the matching rule
    carries a taint condition (a ``none_of`` no source satisfies), so
    rule matching evaluates the label's effective sets too.
    """
    if taint_set_size < 1:
        raise ValueError(f"taint_set_size must be >= 1, got {taint_set_size}")
    rng = random.Random(seed)
    sources = tuple(
        f"src-{rng.getrandbits(32):08x}-{i}" for i in range(taint_set_size)
    )
    policy = Policy(
        name=f"overhead-taint-{taint_set_size}",
        rules=(
            Rule(
                id="taint-inspect",
                when=Selector(
                    tool=BENCH_TOOL,
                    taint=TaintCondition(none_of=("never-a-source",)),
                ),
                effect=Effect(action=Action.ALLOW),
            ),
            _allow_all_rule(),
        ),
    )
    gw = Gateway(policies=[policy])
    gw.register_tool(BENCH_TOOL, ToolTaintSpec.of(adds=("bench-added",)))
    call = ToolCall(
        tool_name=BENCH_TOOL,
        input_label=TaintLabel.of_dimensions(integrity=sources),
    )
    return gw, call


# ---------------------------------------------------------------------------
# Ladder scenarios
# ---------------------------------------------------------------------------


class _NoopAuditWriter:
    """Counts audit calls without paying any I/O cost."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, call: ToolCall, decision: Any) -> None:
        self.calls += 1


def _scenario_raw_call(seed: int) -> Callable[[], Any]:
    fn = _bench_tool
    return lambda: fn(1, 2)


def _scenario_mediated_allow(seed: int) -> Callable[[], Any]:
    gw = Gateway(policies=[make_rule_policy(1, seed=seed)])
    call = ToolCall(tool_name=BENCH_TOOL)
    return lambda: gw.execute(call, _bench_tool, 1, 2)


def _taint_gateway(seed: int, *, track_history: bool, audit: bool) -> Gateway:
    """The shared taint-aware gateway for the upper ladder rungs."""
    policy = Policy(
        name="overhead-ladder",
        rules=(
            Rule(
                id="taint-inspect",
                when=Selector(
                    tool=BENCH_TOOL,
                    taint=TaintCondition(none_of=("never-a-source",)),
                    chain=(
                        ChainCondition(
                            no_prior=(PriorCallMatcher(tool="never-matches-*"),)
                        )
                        if track_history
                        else None
                    ),
                ),
                effect=Effect(action=Action.ALLOW),
            ),
            _allow_all_rule(),
        ),
    )
    gw = Gateway(
        policies=[policy],
        track_history=track_history,
        audit_writer=_NoopAuditWriter() if audit else None,
    )
    gw.register_tool(BENCH_TOOL, ToolTaintSpec.of(adds=("bench-added",)))
    return gw


def _ladder_taint_call(seed: int) -> ToolCall:
    rng = random.Random(seed)
    sources = tuple(
        f"src-{rng.getrandbits(32):08x}-{i}" for i in range(_LADDER_TAINT_SOURCES)
    )
    return ToolCall(
        tool_name=BENCH_TOOL,
        agent_id="bench-agent",
        input_label=TaintLabel.of_dimensions(integrity=sources),
    )


def _scenario_mediated_taint(seed: int) -> Callable[[], Any]:
    gw = _taint_gateway(seed, track_history=False, audit=False)
    call = _ladder_taint_call(seed)
    return lambda: gw.execute(call, _bench_tool, 1, 2)


def _history_scenario(gw: Gateway, call: ToolCall) -> Callable[[], Any]:
    """Execute + reset, so the recorded history length stays constant.

    ``reset_history`` is inside the timed window; its dict-clear cost is
    charged to the rung (see the module docstring).
    """

    def run() -> None:
        gw.execute(call, _bench_tool, 1, 2)
        gw.reset_history()

    return run


def _scenario_mediated_history(seed: int) -> Callable[[], Any]:
    gw = _taint_gateway(seed, track_history=True, audit=False)
    return _history_scenario(gw, _ladder_taint_call(seed))


def _scenario_mediated_full(seed: int) -> Callable[[], Any]:
    gw = _taint_gateway(seed, track_history=True, audit=True)
    return _history_scenario(gw, _ladder_taint_call(seed))


def _scenario_value_ledger(seed: int) -> Callable[[], Any]:
    rng = random.Random(seed)
    ledger = ValueLedger()
    for i in range(LEDGER_PREFILL):
        ledger.record(
            f"prefill-{rng.getrandbits(32):08x}-{i}", TaintLabel.of("web")
        )
    label = TaintLabel.of("web")
    ledger.record("bench-value", label)
    args = {"a": "bench-value", "b": "clean-value"}

    def run() -> None:
        # Re-recording a known value takes the join path; the lookup maps
        # both a known and an unknown argument — the adapter's per-call
        # round-trip (R57).
        ledger.record("bench-value", label)
        ledger.labels_for_args(args)

    return run


_LADDER_TABLE: dict[str, Callable[[int], Callable[[], Any]]] = {
    "raw_call": _scenario_raw_call,
    "mediated_allow": _scenario_mediated_allow,
    "mediated_taint": _scenario_mediated_taint,
    "mediated_history": _scenario_mediated_history,
    "mediated_full": _scenario_mediated_full,
    "value_ledger_roundtrip": _scenario_value_ledger,
}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_ladder(
    *, iterations: int = 5_000, warmup: int = 500, seed: int = 0
) -> list[BenchResult]:
    """Run every ladder rung; results in :data:`LADDER_RUNGS` order."""
    return [
        benchmark(name, _LADDER_TABLE[name](seed), iterations=iterations, warmup=warmup)
        for name in LADDER_RUNGS
    ]


@dataclass(frozen=True)
class SweepPoint:
    """One measured size of one scaling sweep."""

    size: int
    result: BenchResult

    def to_dict(self) -> dict[str, Any]:
        return {"size": self.size, "result": self.result.to_dict()}


def _sweep_decide(gw: Gateway, call: ToolCall) -> Callable[[], Any]:
    return lambda: gw.decide(call)


def run_sweep(
    name: str,
    *,
    sizes: Iterable[int] = DEFAULT_SIZES,
    iterations: int = 2_000,
    warmup: int = 200,
    seed: int = 0,
) -> list[SweepPoint]:
    """Run one scaling sweep (a :data:`SWEEP_NAMES` member) over ``sizes``.

    Each point times the pure :meth:`Gateway.decide` on a freshly built
    seeded workload of that size. Raises ``KeyError`` for an unknown
    sweep name.
    """
    if name == "rule_count":
        def build(size: int) -> tuple[Gateway, ToolCall]:
            gw = Gateway(policies=[make_rule_policy(size, seed=seed)])
            return gw, ToolCall(tool_name=BENCH_TOOL)
    elif name == "history_length":
        def build(size: int) -> tuple[Gateway, ToolCall]:
            return make_history_workload(size, seed=seed)
    elif name == "taint_set_size":
        def build(size: int) -> tuple[Gateway, ToolCall]:
            return make_taint_workload(size, seed=seed)
    else:
        raise KeyError(name)

    points: list[SweepPoint] = []
    for size in sizes:
        gw, call = build(size)
        result = benchmark(
            f"{name}[{size}]",
            _sweep_decide(gw, call),
            iterations=iterations,
            warmup=warmup,
        )
        points.append(SweepPoint(size=size, result=result))
    return points


@dataclass(frozen=True)
class OverheadReport:
    """Everything one suite run measured, plus the pinned run parameters."""

    ladder: tuple[BenchResult, ...]
    sweeps: dict[str, tuple[SweepPoint, ...]]
    iterations: int
    sweep_iterations: int
    warmup: int
    sweep_warmup: int
    seed: int
    python: str
    machine: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": {
                "iterations": self.iterations,
                "sweep_iterations": self.sweep_iterations,
                "warmup": self.warmup,
                "sweep_warmup": self.sweep_warmup,
                "seed": self.seed,
                "python": self.python,
                "machine": self.machine,
            },
            "ladder": [r.to_dict() for r in self.ladder],
            "sweeps": {
                name: [p.to_dict() for p in points]
                for name, points in self.sweeps.items()
            },
        }


def run_overhead_suite(
    *,
    iterations: int = 5_000,
    warmup: int = 500,
    sweep_iterations: int = 2_000,
    sweep_warmup: int = 200,
    sizes: Iterable[int] = DEFAULT_SIZES,
    seed: int = 0,
) -> OverheadReport:
    """Run the ladder and all three sweeps; return one report."""
    sizes = tuple(sizes)
    ladder = run_ladder(iterations=iterations, warmup=warmup, seed=seed)
    sweeps = {
        name: tuple(
            run_sweep(
                name,
                sizes=sizes,
                iterations=sweep_iterations,
                warmup=sweep_warmup,
                seed=seed,
            )
        )
        for name in SWEEP_NAMES
    }
    return OverheadReport(
        ladder=tuple(ladder),
        sweeps=sweeps,
        iterations=iterations,
        sweep_iterations=sweep_iterations,
        warmup=warmup,
        sweep_warmup=sweep_warmup,
        seed=seed,
        python=platform.python_version(),
        machine=f"{platform.system()} {platform.machine()}",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    sep = "  "

    def fmt_row(row: tuple[str, ...]) -> str:
        return sep.join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        )

    lines = [fmt_row(rows[0]), sep.join("-" * w for w in widths)]
    lines.extend(fmt_row(row) for row in rows[1:])
    return "\n".join(lines)


def format_report(report: OverheadReport) -> str:
    """Render the report as fixed-width text tables.

    The ladder table adds an ``overhead_us`` column: each rung's mean
    minus the ``raw_call`` baseline's mean, i.e. what mediation itself
    costs at that rung.
    """
    out: list[str] = []
    out.append(
        f"# gateway overhead (seed={report.seed}, python={report.python}, "
        f"machine={report.machine})"
    )
    out.append("")
    out.append(
        f"## mediation ladder ({report.iterations} iterations, "
        f"{report.warmup} warmup)"
    )
    baseline = report.ladder[0].mean_ns
    rows: list[tuple[str, ...]] = [
        ("rung", "mean_us", "p50_us", "p95_us", "overhead_us")
    ]
    for r in report.ladder:
        rows.append(
            (
                r.name,
                f"{r.mean_ns / 1000:.2f}",
                f"{r.p50_ns / 1000:.2f}",
                f"{r.p95_ns / 1000:.2f}",
                f"{(r.mean_ns - baseline) / 1000:.2f}",
            )
        )
    out.append(_fmt_table(rows))
    for name, points in report.sweeps.items():
        out.append("")
        out.append(
            f"## sweep: {name} ({report.sweep_iterations} iterations, "
            f"{report.sweep_warmup} warmup, Gateway.decide)"
        )
        rows = [("size", "mean_us", "p50_us", "p95_us", "ops/sec")]
        for p in points:
            r = p.result
            rows.append(
                (
                    str(p.size),
                    f"{r.mean_ns / 1000:.2f}",
                    f"{r.p50_ns / 1000:.2f}",
                    f"{r.p95_ns / 1000:.2f}",
                    f"{r.ops_per_sec:,.0f}",
                )
            )
        out.append(_fmt_table(rows))
    return "\n".join(out)


def report_to_json(report: OverheadReport) -> str:
    """The report as a stable JSON document."""
    return json.dumps(report.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agent_policy_gateway.overhead_benchmark",
        description=(
            "Measure per-call gateway mediation overhead and its scaling "
            "against rule count, history length, and taint-set size (R60)."
        ),
    )
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--sweep-iterations", type=int, default=2_000)
    parser.add_argument("--sweep-warmup", type=int, default=200)
    parser.add_argument(
        "--sizes",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated sweep sizes (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny iteration counts and sizes; for smoke tests, not numbers.",
    )
    ns = parser.parse_args(argv)
    if ns.iterations <= 0 or ns.sweep_iterations <= 0:
        parser.error("iteration counts must be > 0")
    if ns.warmup < 0 or ns.sweep_warmup < 0:
        parser.error("warmup counts must be >= 0")
    try:
        ns.size_list = tuple(int(s) for s in ns.sizes.split(",") if s.strip())
    except ValueError:
        parser.error(f"--sizes must be comma-separated integers, got {ns.sizes!r}")
    if not ns.size_list or any(s < 1 for s in ns.size_list):
        parser.error("--sizes needs at least one integer >= 1")
    return ns


def overhead_main(argv: list[str] | None = None) -> int:
    """Console entry point. Returns the process exit code."""
    ns = _parse_args(argv)
    if ns.quick:
        report = run_overhead_suite(
            iterations=50,
            warmup=5,
            sweep_iterations=20,
            sweep_warmup=2,
            sizes=(1, 8, 32),
            seed=ns.seed,
        )
    else:
        report = run_overhead_suite(
            iterations=ns.iterations,
            warmup=ns.warmup,
            sweep_iterations=ns.sweep_iterations,
            sweep_warmup=ns.sweep_warmup,
            sizes=ns.size_list,
            seed=ns.seed,
        )
    out = report_to_json(report) if ns.json else format_report(report)
    sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(overhead_main())
