"""Tests for the R60 gateway-overhead micro-benchmark.

Latency assertions here are deliberately *generous order-of-magnitude
bounds* (the R60 acceptance criterion): the point is to catch a
pathological regression (a hot path suddenly costing milliseconds), not
to pin machine-dependent microsecond values that would flake CI.
Everything else is structural: workload builders really build what the
methodology in ``docs/benchmarks/overhead.md`` says they build.
"""

from __future__ import annotations

import json

import pytest

from agent_policy_gateway.core import ToolCall, Verdict
from agent_policy_gateway.gateway import Gateway
from agent_policy_gateway.overhead_benchmark import (
    BENCH_TOOL,
    DEFAULT_SIZES,
    LADDER_RUNGS,
    SWEEP_NAMES,
    OverheadReport,
    format_report,
    make_history_workload,
    make_rule_policy,
    make_taint_workload,
    overhead_main,
    report_to_json,
    run_ladder,
    run_overhead_suite,
    run_sweep,
)

# Tiny counts keep the whole file fast; bounds below stay valid anyway.
QUICK = dict(iterations=30, warmup=3)
QUICK_SWEEP = dict(iterations=10, warmup=1)

# One millisecond per mediated call of a trivial tool would be a
# pathology on any machine this CI runs on; the observed cost is ~10 µs.
GENEROUS_PER_CALL_NS = 5_000_000  # 5 ms
# The largest quick sweep sizes stay far below this even with the O(n)
# scans they force.
GENEROUS_SWEEP_POINT_NS = 250_000_000  # 250 ms mean per decide


# ---------------------------------------------------------------------------
# Workload builders
# ---------------------------------------------------------------------------


class TestMakeRulePolicy:
    def test_rule_count_and_last_match(self) -> None:
        policy = make_rule_policy(16, seed=0)
        assert len(policy.rules) == 16
        call = ToolCall(tool_name=BENCH_TOOL)
        matched = policy.first_match(call)
        assert matched is not None and matched.id == "match-last"

    def test_single_rule_policy_matches(self) -> None:
        policy = make_rule_policy(1, seed=0)
        assert len(policy.rules) == 1
        assert policy.first_match(ToolCall(tool_name=BENCH_TOOL)) is not None

    def test_miss_rules_never_match_bench_tool(self) -> None:
        policy = make_rule_policy(8, seed=0)
        call = ToolCall(tool_name=BENCH_TOOL)
        for rule in policy.rules[:-1]:
            assert not rule.when.matches(call)

    def test_seeded_determinism(self) -> None:
        a = make_rule_policy(8, seed=7)
        b = make_rule_policy(8, seed=7)
        c = make_rule_policy(8, seed=8)
        assert [r.when.tool for r in a.rules] == [r.when.tool for r in b.rules]
        assert [r.when.tool for r in a.rules] != [r.when.tool for r in c.rules]

    def test_rejects_zero_rules(self) -> None:
        with pytest.raises(ValueError):
            make_rule_policy(0)

    def test_decision_is_allow(self) -> None:
        gw = Gateway(policies=[make_rule_policy(4, seed=0)])
        decision = gw.decide(ToolCall(tool_name=BENCH_TOOL))
        assert decision.verdict == Verdict.ALLOW
        assert decision.rule_id == "match-last"


class TestMakeHistoryWorkload:
    def test_history_has_requested_length(self) -> None:
        gw, call = make_history_workload(12, seed=0)
        assert len(gw.call_history("bench-agent")) == 12

    def test_zero_length_history(self) -> None:
        gw, call = make_history_workload(0, seed=0)
        assert gw.call_history("bench-agent") == ()
        assert gw.decide(call).verdict == Verdict.ALLOW

    def test_decision_pays_the_chain_rule(self) -> None:
        gw, call = make_history_workload(5, seed=0)
        decision = gw.decide(call)
        assert decision.verdict == Verdict.ALLOW
        assert decision.rule_id == "chain-scan"

    def test_entries_carry_seeded_sources(self) -> None:
        gw, _ = make_history_workload(3, seed=0)
        for entry in gw.call_history("bench-agent"):
            assert any(
                s.startswith("src-") for s in entry.output_label.all_sources
            )

    def test_rejects_negative_length(self) -> None:
        with pytest.raises(ValueError):
            make_history_workload(-1)


class TestMakeTaintWorkload:
    def test_label_has_requested_size(self) -> None:
        _, call = make_taint_workload(9, seed=0)
        assert len(call.input_label.integrity_sources) == 9

    def test_output_label_joins_spec_add(self) -> None:
        gw, call = make_taint_workload(4, seed=0)
        decision = gw.decide(call)
        assert decision.verdict == Verdict.ALLOW
        assert decision.rule_id == "taint-inspect"
        # 4 input sources + the spec's one added source.
        assert len(decision.output_label.all_sources) == 5
        assert "bench-added" in decision.output_label.all_sources

    def test_seeded_determinism(self) -> None:
        _, a = make_taint_workload(6, seed=3)
        _, b = make_taint_workload(6, seed=3)
        _, c = make_taint_workload(6, seed=4)
        assert a.input_label == b.input_label
        assert a.input_label != c.input_label

    def test_rejects_zero_sources(self) -> None:
        with pytest.raises(ValueError):
            make_taint_workload(0)


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


class TestLadder:
    def test_rung_names_and_order(self) -> None:
        results = run_ladder(**QUICK)
        assert tuple(r.name for r in results) == LADDER_RUNGS

    def test_all_rungs_positive_and_bounded(self) -> None:
        results = run_ladder(**QUICK)
        for r in results:
            assert r.mean_ns > 0
            assert r.mean_ns < GENEROUS_PER_CALL_NS, r.name

    def test_mediation_costs_more_than_raw(self) -> None:
        # Direction only — the one comparison that holds on any machine.
        results = {r.name: r for r in run_ladder(iterations=200, warmup=20)}
        assert results["mediated_allow"].mean_ns > results["raw_call"].mean_ns


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


class TestSweeps:
    @pytest.mark.parametrize("name", SWEEP_NAMES)
    def test_sizes_and_bounds(self, name: str) -> None:
        points = run_sweep(name, sizes=(1, 8, 32), **QUICK_SWEEP)
        assert [p.size for p in points] == [1, 8, 32]
        for p in points:
            assert p.result.mean_ns > 0
            assert p.result.mean_ns < GENEROUS_SWEEP_POINT_NS
            assert p.result.name == f"{name}[{p.size}]"

    def test_history_sweep_accepts_size_zero(self) -> None:
        points = run_sweep("history_length", sizes=(0, 4), **QUICK_SWEEP)
        assert [p.size for p in points] == [0, 4]

    def test_unknown_sweep_raises(self) -> None:
        with pytest.raises(KeyError):
            run_sweep("nope", sizes=(1,), **QUICK_SWEEP)


# ---------------------------------------------------------------------------
# Suite, report, CLI
# ---------------------------------------------------------------------------


def _quick_suite() -> OverheadReport:
    return run_overhead_suite(
        iterations=30,
        warmup=3,
        sweep_iterations=10,
        sweep_warmup=1,
        sizes=(1, 8),
        seed=0,
    )


class TestReport:
    def test_suite_shape(self) -> None:
        report = _quick_suite()
        assert tuple(r.name for r in report.ladder) == LADDER_RUNGS
        assert set(report.sweeps) == set(SWEEP_NAMES)
        for points in report.sweeps.values():
            assert [p.size for p in points] == [1, 8]
        assert report.seed == 0
        assert report.python  # pinned run parameters present
        assert report.machine

    def test_json_round_trip(self) -> None:
        report = _quick_suite()
        doc = json.loads(report_to_json(report))
        assert doc["params"]["seed"] == 0
        assert [r["name"] for r in doc["ladder"]] == list(LADDER_RUNGS)
        assert set(doc["sweeps"]) == set(SWEEP_NAMES)

    def test_text_report_mentions_everything(self) -> None:
        text = format_report(_quick_suite())
        for rung in LADDER_RUNGS:
            assert rung in text
        for sweep in SWEEP_NAMES:
            assert f"sweep: {sweep}" in text
        assert "overhead_us" in text

    def test_default_sizes_are_ascending(self) -> None:
        assert list(DEFAULT_SIZES) == sorted(set(DEFAULT_SIZES))


class TestCli:
    def test_quick_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert overhead_main(["--quick"]) == 0
        out = capsys.readouterr().out
        assert "mediation ladder" in out
        assert "sweep: rule_count" in out

    def test_quick_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert overhead_main(["--quick", "--json"]) == 0
        doc = json.loads(capsys.readouterr().out)
        assert set(doc["sweeps"]) == set(SWEEP_NAMES)

    def test_bad_sizes_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc:
            overhead_main(["--sizes", "1,frog"])
        assert exc.value.code == 2

    def test_zero_iterations_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc:
            overhead_main(["--iterations", "0"])
        assert exc.value.code == 2
