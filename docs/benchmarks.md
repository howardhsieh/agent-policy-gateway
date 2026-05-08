# Benchmarks

The `agent-policy-gateway` ships with a small benchmark harness that
measures per-call overhead and throughput on the gateway's hot paths.
The goal is *reproducible, comparable* numbers across releases — not a
competition-grade microbench framework — so the suite is intentionally
short, dependency-free, and easy to run anywhere `pip install -e .`
already works.

## Running the suite

After installing the package (`pip install -e .` is enough; no extras
required), invoke the console script:

```
apg-bench
```

This runs every scenario in [`DEFAULT_SCENARIOS`](#scenarios) with
5,000 timed iterations and 500 warmup iterations apiece and prints a
fixed-width text table to stdout. For machine-readable output:

```
apg-bench --json
```

Common flags:

| Flag                   | Effect                                                |
|------------------------|-------------------------------------------------------|
| `--iterations N`       | Timed calls per scenario (default 5000).              |
| `--warmup N`           | Un-timed warmup calls before measurement (default 500). |
| `--scenario NAME`      | Run only NAME; pass multiple times for a subset.      |
| `--json`               | Emit JSON instead of a table.                         |

Unknown scenario names exit with code 2 (argparse default), so the
script is safe to wire into CI without further error-handling.

## Scenarios

Four scenarios span the gateway's hot paths so overhead and throughput
can be compared apples-to-apples:

- **`raw_call`** — the underlying tool function called directly, with
  no gateway and no audit. Establishes a baseline so the gateway-induced
  overhead is observable as a *delta* rather than an absolute number.
- **`gateway_allow`** — `Gateway.execute` plus `Gateway.wrap_tool`
  against a policy whose first matching rule is an `allow`. Exercises
  the decide-then-call hot path that production callers see most often.
- **`gateway_deny`** — same wrapper, against a policy whose first
  matching rule is a `deny`. The wrapped callable raises `PolicyDenied`
  *before* the underlying tool runs; the timed callable swallows the
  exception so percentiles reflect the refusal cost itself, not whatever
  the harness does after the throw.
- **`gateway_allow_audit`** — gateway-allow with an in-process no-op
  `AuditWriter` attached. Isolates the audit-call dispatch cost from the
  JSON-encode + fsync cost that `JsonlAuditWriter` adds on top.

## Programmatic API

The harness is a thin wrapper around three public helpers, all
re-exported from `agent_policy_gateway`:

```python
from agent_policy_gateway import (
    benchmark,
    run_default_suite,
    format_results_table,
    results_to_json,
)

results = run_default_suite(iterations=5000, warmup=500)
print(format_results_table(results))

# Or measure something custom:
r = benchmark("hello", lambda: "world", iterations=10000, warmup=500)
print(r.ops_per_sec, r.p99_ns)
```

Each `BenchResult` is a frozen dataclass with `name`, `iterations`,
`warmup`, `total_ns`, `mean_ns`, `min_ns`, `max_ns`, the
50/95/99-th percentiles in nanoseconds, and a derived `ops_per_sec`.
`to_dict()` returns the same fields as a plain mapping for callers
that want to write their own JSON.

## Methodology notes

- Latency is recorded with `time.perf_counter_ns()` — one read before
  and after each call — so percentiles reflect real per-call latency
  rather than wall-clock-divided averages.
- The trivial tool function (`x + y`) is held constant across scenarios
  so the delta between `raw_call` and `gateway_allow` is mostly
  attributable to the gateway, not to the tool body.
- Percentiles use nearest-rank with ceil rounding; a 1-sample input
  always yields the sole sample. This is deliberately simple — fine for
  monitoring overhead changes, not for headline statistics — and is
  documented in `bench._percentile_ns`.
- The audit scenario uses an in-process no-op writer so `gateway_allow`
  vs `gateway_allow_audit` measures the audit-dispatch cost only;
  `JsonlAuditWriter` is intentionally *not* part of the default suite
  because its fsync+JSON cost would dominate everything else and make
  cross-host comparisons noisy. Add a custom scenario via `benchmark()`
  if you want to measure on-disk audit performance.
