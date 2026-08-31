# Long-horizon stateful eval demo (R55)

Replays the "laundering adversary" scenario family as multi-turn
**persistent sessions** and shows the one result the R55 harness exists to
make visible: over a long horizon, a stateless input-taint defense and a
chain-history defense — which look identical on a single call — come apart.

```bash
python -m examples.stateful
```

prints the three-arm summary table and a pass/fail line per R55 invariant,
exiting non-zero if any fails (so it doubles as a CI check).

## What each scenario does

Every scenario is one persistent session:

1. **read** an untrusted reader (taints the session, R49a),
2. `k` turns of benign **work** (the horizon),
3. optionally a **sanitize** turn that declassifies the session label (an
   R52 grant both defended policies share),
4. **act** on an external sink — as a legitimate user call (measures
   utility) or an injected attack call (measures compromise).

## The three arms

| arm | policy | rule |
|-----|--------|------|
| `no-defense` | — | bare runtime; every sink executes |
| `apg-input-taint` | [`policies/stateful-input-taint.yaml`](../../policies/stateful-input-taint.yaml) | deny a sink when the *current* session label is untrusted |
| `apg-chain` | [`policies/stateful-chain.yaml`](../../policies/stateful-chain.yaml) | deny a sink when the session *history* records an untrusted read |

## The finding

The `sanitize` declassify that lets `apg-input-taint` recover utility on a
legitimate laundered flow **equally launders the attack**: its compromise
rate on laundered scenarios jumps back to 100%. `apg-chain` decides the
same sinks over the immutable call history, so it holds at 0% compromise
across the whole horizon — at the cost of also blocking legitimate
post-read sinks. That utility/robustness split is the input to R56.

The full write-up, with the numbers, is in
[`docs/benchmarks/stateful.md`](../../docs/benchmarks/stateful.md); the
harness itself is `agent_policy_gateway.stateful_eval` and the benchmark is
`agent_policy_gateway.stateful_benchmark`.
