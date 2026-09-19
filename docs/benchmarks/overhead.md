# Gateway overhead micro-benchmark (R60)

Every other document in this directory measures what the gateway
*prevents*; this one measures what mediation *costs*. Two questions:

1. **Per-call overhead** — what does routing one tool call through
   `Gateway.execute` cost, and which bookkeeping feature (policy
   evaluation, taint propagation, history tracking, per-value ledger)
   pays for what share of it?
2. **Scaling** — how does the pure decision cost (`Gateway.decide`)
   grow with rule count, session-history length, and taint-set size?

## Methodology (pinned)

- **Command:**

  ```bash
  python -m agent_policy_gateway.overhead_benchmark --seed 0
  ```

  (defaults: 5,000 timed iterations / 500 warmup per ladder rung;
  2,000 / 200 per sweep point; sweep sizes 1, 16, 64, 256, 1024).
- **Timer:** the R12 `bench.benchmark` harness — `time.perf_counter_ns`
  around each call, warmup un-timed, nearest-rank percentiles. No
  third-party benchmark framework.
- **Workloads:** synthetic and seeded (`random.Random(seed)` generates
  every rule tool-name and taint source), so the same seed measures a
  byte-identical workload. The mediated tool is a trivial pure-Python
  add, so the numbers isolate the gateway, not the tool.
- **Worst cases by construction:** the rule-count sweep matches only
  the *last* of *N* rules (full first-match scan); the history sweep's
  chain rule carries a `no_prior` matcher that can never match, forcing
  the full O(H) history walk; the taint sweep's rule carries a taint
  condition, so the label's effective sets are computed during matching
  *and* joined by `propagate`.
- **Ladder history rung:** the gateway's history is reset (public
  `reset_history()`) inside the timed window after every call so the
  recorded length stays constant; the rung therefore slightly
  *overstates* history cost — the conservative direction.
- **Environment for the pinned numbers below:** Python 3.11.15, Linux
  x86_64, Intel Xeon @ 2.10 GHz (4 vCPU cloud container), package
  installed with `pip install -e .`. Run-to-run drift on this machine
  is ~5% on means; treat absolute values as machine-specific and the
  *deltas and growth curves* as the finding.

## Per-call mediation ladder

Each rung adds one feature; `overhead_us` is the rung's mean minus the
`raw_call` baseline mean (seed 0, 2026-09-19):

| rung | mean µs | p50 µs | p95 µs | overhead µs |
|------|--------:|-------:|-------:|------------:|
| `raw_call` (no gateway) | 0.12 | 0.11 | 0.14 | 0.00 |
| `mediated_allow` (1 allow rule) | 12.20 | 11.54 | 16.42 | 12.08 |
| `mediated_taint` (+ 4-source label, spec add, taint rule) | 10.68 | 10.13 | 13.64 | 10.56 |
| `mediated_history` (+ `track_history`, chain rule) | 13.15 | 12.50 | 16.77 | 13.03 |
| `mediated_full` (+ no-op audit writer) | 12.96 | 12.35 | 17.04 | 12.85 |
| `value_ledger_roundtrip` (R57 record + 2-arg lookup, 1,024-entry ledger) | 4.20 | 3.93 | 4.82 | 4.08 |

**Reading.** Mediating a call costs **~11–13 µs** on this machine —
about **80,000 mediated calls per second** on one core. The rungs sit
within run-to-run noise of each other (`mediated_taint` even reads
below `mediated_allow` here): at realistic label/history sizes the cost
is dominated by the **fixed decide path** (selector matching, label
canonicalization, `Decision` construction), not by any individual
bookkeeping feature. Taint, history, and audit dispatch are each ≲1–2 µs
riders on that fixed cost. The R57 value-ledger round-trip is an
independent ~4 µs paid by adapters that enable `track_values`. Against
a tool call that does *any* real work — a subprocess, a network round
trip (milliseconds to seconds), or an LLM turn (seconds) — mediation
overhead is three to six orders of magnitude below the work being
mediated.

## Scaling sweeps (`Gateway.decide`, pure decision)

### Rule count (only the last of *N* rules matches)

| rules | mean µs | p50 µs | p95 µs | ops/sec |
|------:|--------:|-------:|-------:|--------:|
| 1 | 11.31 | 10.76 | 15.49 | 88,397 |
| 16 | 19.04 | 18.31 | 24.34 | 52,509 |
| 64 | 41.91 | 39.85 | 57.70 | 23,860 |
| 256 | 128.80 | 125.96 | 148.29 | 7,764 |
| 1024 | 485.89 | 483.94 | 515.04 | 2,058 |

Linear, as first-match evaluation must be: ~**0.46 µs per scanned
non-matching rule** on top of the ~11 µs fixed cost. Even a
1,024-rule worst-case policy decides in under half a millisecond;
policies in this repository are 3–20 rules.

### History length (chain rule walks all *H* entries)

| entries | mean µs | p50 µs | p95 µs | ops/sec |
|--------:|--------:|-------:|-------:|--------:|
| 1 | 13.03 | 12.47 | 14.91 | 76,735 |
| 16 | 20.66 | 19.86 | 26.24 | 48,397 |
| 64 | 44.68 | 42.86 | 58.88 | 22,382 |
| 256 | 133.40 | 130.38 | 150.03 | 7,497 |
| 1024 | 502.11 | 500.29 | 524.79 | 1,992 |

Also linear — ~**0.48 µs per recorded entry** for a
history-referencing rule (the per-decide history-view copy plus one
`fnmatch` per entry). A 1,000-call session still decides in ~0.5 ms.
Sessions in the R55/R59 long-horizon benchmarks record 6–30 calls.

### Taint-set size (*S* integrity sources joined and inspected)

| sources | mean µs | p50 µs | p95 µs | ops/sec |
|--------:|--------:|-------:|-------:|--------:|
| 1 | 10.11 | 9.71 | 11.19 | 98,890 |
| 16 | 10.81 | 10.38 | 11.99 | 92,475 |
| 64 | 13.03 | 12.50 | 14.82 | 76,758 |
| 256 | 21.78 | 21.01 | 27.11 | 45,912 |
| 1024 | 60.30 | 58.01 | 76.16 | 16,582 |

The flattest axis: frozenset unions and subset checks cost
~**0.05 µs per source** — an order of magnitude cheaper per item than
a rule or history entry, because it is C-level set algebra rather than
per-item Python matching. Real labels carry ones-to-tens of sources
(the R56 family's largest label has 3).

## Summary for the write-up (R61)

- Full mediation (policy + dual-label taint + history + audit
  dispatch) costs **~12 µs per call** (~80 k calls/s/core) on
  commodity cloud hardware; the fixed decide path dominates and each
  feature adds ≲1–2 µs at realistic sizes.
- Cost grows **linearly** in worst-case scanned rules (~0.46 µs/rule)
  and in chain-rule history length (~0.48 µs/entry), and stays
  **near-flat** in taint-set size (~0.05 µs/source).
- At every size the R55/R56/R59 benchmarks actually use (≤20 rules,
  ≤30 history entries, ≤4 sources), per-call overhead stays **under
  ~25 µs** — negligible against any tool that performs I/O and
  invisible against an LLM turn.

## Regenerating

```bash
python -m agent_policy_gateway.overhead_benchmark --seed 0          # this table
python -m agent_policy_gateway.overhead_benchmark --seed 0 --json  # machine-readable
python -m agent_policy_gateway.overhead_benchmark --quick          # smoke run
```

CI keeps only generous order-of-magnitude bounds on these paths
(`tests/test_overhead_benchmark.py`), so the suite stays stable across
machines; the pinned numbers above are refreshed manually by rerunning
the command on a quiet machine and updating this file.
