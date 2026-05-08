"""Tests for the benchmark harness.

These tests do *not* assert specific timing thresholds (latencies
depend on the host); they assert structural and contract invariants
that hold on any machine: percentile ordering, scenario semantics,
table/JSON shape, and CLI behaviour.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from agent_policy_gateway.bench import (
    DEFAULT_SCENARIOS,
    BenchResult,
    _NoopAuditWriter,
    _percentile_ns,
    bench_main,
    benchmark,
    format_results_table,
    make_scenario,
    results_to_json,
    run_default_suite,
    scenario_gateway_allow,
    scenario_gateway_allow_audit,
    scenario_gateway_deny,
    scenario_raw_call,
)
from agent_policy_gateway.gateway import PolicyDenied

# ---------------------------------------------------------------------------
# benchmark()
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_runs_iterations_and_warmup(self) -> None:
        calls = [0]

        def fn() -> None:
            calls[0] += 1

        r = benchmark("noop", fn, iterations=20, warmup=5)
        assert calls[0] == 25
        assert r.name == "noop"
        assert r.iterations == 20
        assert r.warmup == 5

    def test_default_warmup_is_zero(self) -> None:
        calls = [0]

        def fn() -> None:
            calls[0] += 1

        benchmark("noop", fn, iterations=10)
        assert calls[0] == 10

    def test_percentile_ordering_invariant(self) -> None:
        r = benchmark("noop", lambda: None, iterations=200, warmup=10)
        # min <= p50 <= p95 <= p99 <= max — always true regardless of host.
        assert r.min_ns <= r.p50_ns <= r.p95_ns <= r.p99_ns <= r.max_ns

    def test_total_and_mean_are_consistent(self) -> None:
        r = benchmark("noop", lambda: None, iterations=100, warmup=0)
        # total / iterations == mean (within float epsilon).
        assert r.mean_ns == pytest.approx(r.total_ns / r.iterations)

    def test_ops_per_sec_positive(self) -> None:
        r = benchmark("noop", lambda: None, iterations=100, warmup=0)
        assert r.ops_per_sec > 0

    def test_zero_iterations_rejected(self) -> None:
        with pytest.raises(ValueError, match="iterations must be > 0"):
            benchmark("noop", lambda: None, iterations=0)

    def test_negative_warmup_rejected(self) -> None:
        with pytest.raises(ValueError, match="warmup must be >= 0"):
            benchmark("noop", lambda: None, iterations=10, warmup=-1)

    def test_to_dict_shape(self) -> None:
        r = benchmark("noop", lambda: None, iterations=10, warmup=0)
        d = r.to_dict()
        assert set(d) == {
            "name",
            "iterations",
            "warmup",
            "total_ns",
            "mean_ns",
            "min_ns",
            "max_ns",
            "p50_ns",
            "p95_ns",
            "p99_ns",
            "ops_per_sec",
        }
        assert d["name"] == "noop"
        assert d["iterations"] == 10


# ---------------------------------------------------------------------------
# _percentile_ns
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_single_sample_returns_self(self) -> None:
        assert _percentile_ns([42], 50) == 42
        assert _percentile_ns([42], 99) == 42
        assert _percentile_ns([42], 0) == 42

    def test_known_percentiles(self) -> None:
        # 1..100; nearest-rank ceil-rounded.
        samples = list(range(1, 101))
        assert _percentile_ns(samples, 50) == 50
        assert _percentile_ns(samples, 95) == 95
        assert _percentile_ns(samples, 99) == 99
        assert _percentile_ns(samples, 100) == 100

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _percentile_ns([], 50)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            _percentile_ns([1, 2, 3], -1)
        with pytest.raises(ValueError):
            _percentile_ns([1, 2, 3], 101)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_raw_call_returns_three(self) -> None:
        # _bench_tool(1, 2) == 3
        assert scenario_raw_call()() == 3

    def test_gateway_allow_returns_three(self) -> None:
        assert scenario_gateway_allow()() == 3

    def test_gateway_deny_swallows_policy_denied(self) -> None:
        # The benchmark-facing callable must NOT raise — that would
        # break the timing loop. PolicyDenied is caught inside.
        fn = scenario_gateway_deny()
        # Should not raise.
        result = fn()
        assert result is None

    def test_gateway_deny_actually_denies_when_unwrapped(self) -> None:
        # Sanity-check that the deny scenario is wired to a deny policy:
        # build a fresh deny scenario, but verify its underlying contract
        # by directly invoking the gateway-wrapped tool inside.
        # We re-construct via gateway_allow vs gateway_deny: deny must
        # raise when called without the swallowing wrapper.
        from agent_policy_gateway.bench import _ALLOW_POLICY_YAML, _DENY_POLICY_YAML, _bench_tool
        from agent_policy_gateway.gateway import Gateway
        from agent_policy_gateway.policy import load_policy_str

        deny_gw = Gateway(policies=[load_policy_str(_DENY_POLICY_YAML)])
        wrapped = deny_gw.wrap_tool(_bench_tool, tool_name="bench_tool")
        with pytest.raises(PolicyDenied):
            wrapped(1, 2)

        allow_gw = Gateway(policies=[load_policy_str(_ALLOW_POLICY_YAML)])
        wrapped_allow = allow_gw.wrap_tool(_bench_tool, tool_name="bench_tool")
        assert wrapped_allow(1, 2) == 3

    def test_gateway_allow_audit_invokes_writer(self) -> None:
        # The scenario factory closes over its own _NoopAuditWriter; we
        # can't reach it, but we can build the same setup and verify the
        # writer is invoked. The factory itself is just exercised here
        # to make sure it doesn't raise on construction or on call.
        fn = scenario_gateway_allow_audit()
        assert fn() == 3
        # And confirm the wiring directly:
        from agent_policy_gateway.bench import _ALLOW_POLICY_YAML, _bench_tool
        from agent_policy_gateway.gateway import Gateway
        from agent_policy_gateway.policy import load_policy_str

        audit = _NoopAuditWriter()
        gw = Gateway(policies=[load_policy_str(_ALLOW_POLICY_YAML)], audit_writer=audit)
        wrapped = gw.wrap_tool(_bench_tool, tool_name="bench_tool")
        wrapped(1, 2)
        wrapped(1, 2)
        assert audit.calls == 2

    def test_make_scenario_known(self) -> None:
        for name in DEFAULT_SCENARIOS:
            assert callable(make_scenario(name))

    def test_make_scenario_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            make_scenario("does-not-exist")


# ---------------------------------------------------------------------------
# Suite + reporting
# ---------------------------------------------------------------------------


class TestSuite:
    def test_run_default_suite_returns_one_per_scenario(self) -> None:
        results = run_default_suite(iterations=20, warmup=5)
        assert len(results) == len(DEFAULT_SCENARIOS)
        assert [r.name for r in results] == list(DEFAULT_SCENARIOS)
        for r in results:
            assert isinstance(r, BenchResult)
            assert r.iterations == 20
            assert r.warmup == 5

    def test_run_default_suite_subset(self) -> None:
        results = run_default_suite(
            iterations=10, warmup=0, scenarios=("raw_call", "gateway_allow")
        )
        assert [r.name for r in results] == ["raw_call", "gateway_allow"]

    def test_format_results_table_has_headers_and_rows(self) -> None:
        results = run_default_suite(iterations=10, warmup=0)
        out = format_results_table(results)
        # Header row.
        assert "scenario" in out
        assert "ops/sec" in out
        # One line per scenario.
        for r in results:
            assert r.name in out
        # Separator line uses dashes.
        assert "---" in out

    def test_results_to_json_roundtrips(self) -> None:
        results = run_default_suite(iterations=10, warmup=0)
        doc = json.loads(results_to_json(results))
        assert "results" in doc
        assert len(doc["results"]) == len(results)
        first = doc["results"][0]
        assert first["name"] == DEFAULT_SCENARIOS[0]
        assert {"mean_ns", "p95_ns", "ops_per_sec"} <= set(first)

    def test_format_table_empty_results_is_just_headers(self) -> None:
        out = format_results_table([])
        assert out.splitlines()[0].split() == [
            "scenario",
            "iters",
            "mean_us",
            "p50_us",
            "p95_us",
            "p99_us",
            "ops/sec",
        ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_default_invocation_prints_table(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = bench_main(["--iterations", "10", "--warmup", "0"])
        assert rc == 0
        out = buf.getvalue()
        assert "scenario" in out
        for name in DEFAULT_SCENARIOS:
            assert name in out

    def test_json_flag_prints_json(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = bench_main(["--iterations", "10", "--warmup", "0", "--json"])
        assert rc == 0
        doc = json.loads(buf.getvalue())
        assert isinstance(doc["results"], list)
        assert len(doc["results"]) == len(DEFAULT_SCENARIOS)

    def test_scenario_flag_filters(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = bench_main(
                [
                    "--iterations",
                    "10",
                    "--warmup",
                    "0",
                    "--scenario",
                    "raw_call",
                    "--json",
                ]
            )
        assert rc == 0
        doc = json.loads(buf.getvalue())
        assert [r["name"] for r in doc["results"]] == ["raw_call"]

    def test_unknown_scenario_exits_2(self) -> None:
        # argparse choices=DEFAULT_SCENARIOS -> SystemExit(2) on unknown.
        with pytest.raises(SystemExit) as exc:
            bench_main(["--scenario", "does-not-exist"])
        assert exc.value.code == 2

    def test_zero_iterations_exits_2(self) -> None:
        with pytest.raises(SystemExit) as exc:
            bench_main(["--iterations", "0"])
        assert exc.value.code == 2

    def test_negative_warmup_exits_2(self) -> None:
        with pytest.raises(SystemExit) as exc:
            bench_main(["--warmup", "-1"])
        assert exc.value.code == 2
