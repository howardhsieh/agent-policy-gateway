# Long-horizon stateful adversarial eval (R55)

The [AgentDojo benchmark](agentdojo.md) (R50) replays one episode per
(user task, injection task) pair and **resets the session between
episodes**. That isolates the policy's per-call decision, but by
construction it cannot see what a long-lived agent session actually
exposes: attacker influence that **persists and compounds across many
turns**. This page reports the R55 harness, which drives *scenarios* — an
ordered sequence of turns replayed through **one persistent runtime**,
with the session taint (R49a) and the gateway's call history (R53) carried
across the whole horizon and reset only once at the start.

That single change — never resetting between turns — is the whole point of
the harness. The state a defense accumulates is exactly the state a
long-horizon adversary tries to outlast or launder.

## The question this benchmark answers

Phase 2 built two ways to gate a sink on untrusted input: an **input-taint**
rule on the current session label (R51), and a **chain-history** rule over
the session's recorded calls (R53). On the single-episode R50 replay their
numbers coincide, because the session label there only ever accumulates.
R53's design note flagged the difference that a longer horizon would
expose: a chain rule *"still fires after a declassify grant strips the
taint."* R55 is the harness that measures it.

The scenario family is a **laundering adversary**. Every scenario:

1. reads attacker-authorable content (a turn that taints the session),
2. does `k` turns of benign work (the horizon, `k ∈ {1,2,3,4}`), then
3. acts on an external sink — as a **legitimate user call** (the *benign*
   family, measuring utility) or as an **injected attack call** (the
   *armed* family, measuring compromise).

Each post-read scenario comes in two variants: **direct** (the sink follows
straight through) and **launder** (a vetted `sanitize` turn declassifies
the session label first, via an R52 grant both defended policies share). A
third *clean* benign family has no untrusted read at all — a sink no
defense should block, so utility rewards precision instead of collapsing to
zero.

## Headline numbers

Three arms over the same 60 scenarios — 24 armed (attack sink), 24 benign
post-read, 12 clean benign — each replayed as one persistent session:

```text
arm              scenarios  utility compromise  direct launder refusals persist
-------------------------------------------------------------------------------
no-defense              60   100.0%     100.0%  100.0%  100.0%        0    0.00
apg-input-taint         60    66.7%      50.0%    0.0%  100.0%       24    3.20
apg-chain               60    33.3%       0.0%    0.0%    0.0%       48    3.20
```

* **utility** — share of the 36 benign scenarios (clean + post-read) whose
  legitimate sink executed.
* **compromise** — share of the 24 armed scenarios where the attack sink
  executed at least once (the stateful analogue of R50's ASR).
* **direct / launder** — compromise split across the two armed variants.
* **persist** — mean `taint_persistence`: how many turns, on average, the
  untrusted source stayed live in the session label. (The `no-defense`
  arm has no taint tracking, so it reads 0.)

The load-bearing row is the split:

* **A mid-session declassify launders the attack.** `apg-input-taint` stops
  every **direct** attack (0%), but on **laundered** scenarios its
  compromise rate jumps straight back to the no-defense level (**100%**):
  once the `sanitize` step clears the session label, the input-taint rule
  can no longer tell a laundered *attack* sink call from a legitimately
  recovered *user* one — both arrive with a clean label. The very grant
  that lets the arm recover utility on benign laundered flows is what
  reopens the attack surface.
* **Chain-history survives the declassify.** `apg-chain` holds at **0%**
  compromise on *both* variants across the whole horizon: the history entry
  recording the earlier untrusted read is immutable, so a `sanitize` step
  that strips the live label changes nothing. This is exactly the
  robustness R53 promised, now measured over a long session rather than
  argued from a single call.

## The utility/robustness split (the input to R56)

The two defended arms are not one strictly better than the other — they
trade different things, and the harness quantifies the trade:

| arm             | utility | compromise | what it buys |
|-----------------|--------:|-----------:|--------------|
| apg-input-taint |   66.7% |      50.0% | recovers laundered legitimate flows — and launders attacks with them |
| apg-chain       |   33.3% |       0.0% | robust across the horizon — at the cost of every legitimate post-read sink |

`apg-input-taint` keeps more utility (66.7% vs 33.3%) because the declassify
lets legitimate post-read sinks through; `apg-chain` is strictly more secure
(0% vs 50% compromise) because it ignores the declassify entirely. Neither
is the endpoint: the chain rule here is a blanket *"no sink after any
untrusted read"*, which is why it also blocks the benign post-read family.
Making it selective — conditioning on `resource`, on the R30 provenance
chain, or on *which* prior source is present — is what **R56** measures
across APG, Progent, and Fides.

## What is measured (and what is not)

Like the R50 benchmark this is a **deterministic scripted replay**, not an
LLM-driven run — there is no model deciding whether to follow an injection.
Unlike R50 it needs **no `agentdojo` package and no API keys**: the family
runs the real [`Gateway`](../design.md) over a tiny in-process runtime, so
the numbers above are reproduced by the ordinary test suite
(`tests/test_stateful_eval.py` pins every rate).

The harness itself is framework-agnostic. `scenario_from_suite` composes a
long-horizon scenario from real AgentDojo suite tasks — several user tasks
as turns, then an injection task's calls as the attack turn — so the same
persistent-session machinery runs against the real suites through a
`gate_suite` runtime (an integration test builds one from banking and
confirms the turn-1 read's taint is still live at the final turn). A
model-in-the-loop long-horizon eval — where injections succeed only
sometimes and the agent chooses its own actions — remains future work; R55
is the *stateful* dimension, deterministic and free.

### Metrics on each scenario report

* **first_compromise_turn** — 1-based index of the earliest turn an
  attacker call executed (`None` if the defense held the whole horizon).
  *When*, not just whether, the session was breached.
* **taint_persistence** — the longest span, in turns, that any one source
  stayed live in the session label (first appearance through last). Under a
  monotonically accumulating label this is "turns from the read to the
  end"; under a mid-session declassify it correctly shrinks to the window
  the source was actually live — the window a laundering attack opens.
* per-turn **taint snapshots**, task success, and refusals — a full
  transcript of the session, not a single verdict.

## Reproducing

From the repository root (no extras required):

```bash
python -m agent_policy_gateway.stateful_benchmark
```

That replays all three arms over the 60-scenario family and prints the
table above. Options:

```bash
# machine-readable per-arm summaries (including the armed/benign breakdowns)
python -m agent_policy_gateway.stateful_benchmark --json

# resolve the arm policy files under a different directory
python -m agent_policy_gateway.stateful_benchmark --policy-dir /path/to/repo
```

The worked example prints the same table and then checks the R55
invariants, exiting non-zero if any fails:

```bash
python -m examples.stateful
```

Exit codes for the benchmark: **0** success, **2** an arm policy file could
not be found under `--policy-dir`.

## Relation to the other benchmarks

The [overhead page](../benchmarks.md) measures per-call latency; the
[AgentDojo page](agentdojo.md) measures single-episode security value on a
public benchmark. This page measures what a defense retains **across a long
stateful session** — the axis on which input-taint and chain-history, which
look identical on a single call, come apart.
