# AgentDojo: no defense vs. APG (R50)

This page reports the first security benchmark of the gateway: the
[AgentDojo](https://github.com/ethz-spylab/agentdojo) prompt-injection
benchmark's four default task suites, replayed through the R49 adapter
stack under **(a)** no defense and **(b)** the shipped
[`policies/agentdojo.yaml`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/policies/agentdojo.yaml)
taint policy, comparing **attack success rate** and **task utility**
between the two arms.

## Headline numbers

AgentDojo `v1.2.1` suites via `agentdojo` 0.1.35, scripted ground-truth
replay (see [What is measured](#what-is-measured)); every
user&nbsp;task&nbsp;×&nbsp;injection&nbsp;task pair of each suite, both arms:

```text
suite      arm        episodes  utility  armed ASR(sink) ASR(any-call) refusals
-------------------------------------------------------------------------------
banking    no-defense      144   100.0%    144    100.0%        100.0%        0
banking    apg             144    25.0%    144      0.0%         11.1%      284
slack      no-defense      105   100.0%     84    100.0%        100.0%        0
slack      apg             105     4.8%     84      0.0%         60.0%      296
travel     no-defense      140   100.0%    120    100.0%        100.0%        0
travel     apg             140    70.0%    120      0.0%         50.0%      162
workspace  no-defense      560   100.0%    240    100.0%        100.0%        0
workspace  apg             560    55.0%    240      0.0%         50.0%      560
```

Two things to read off the table:

* **The policy stops every measurable attack.** Sink-level ASR goes from
  100% to **0%** on all four suites: with the session label carrying
  `agentdojo:untrusted` after any injectable read, no attacker-directed
  call on a classified external sink gets through.
* **The utility cost is the whole story of Phase 2.** The same blanket
  rule ("no sink call once the session is tainted") refuses legitimate
  user-task sink calls that follow an injectable read, and the cost
  varies enormously by suite: travel keeps 70% utility, banking 25%,
  slack collapses to 4.8% (almost every slack task reads a webpage,
  channel, or inbox before acting). This is exactly the tension the
  finer-grained mechanisms on the roadmap — dual-label taint (R51),
  declarative declassify (R52), chain-level policies (R53) — exist to
  relax without reopening the attack surface.

## What is measured

The benchmark is a **scripted ground-truth replay**, not an LLM-driven
run. For every (user task, injection task) pair, the episode script is
the user task's ground-truth tool calls (what a faithful agent would do)
followed by the injection task's ground-truth calls (what a fully
hijacked agent would attempt) — the pessimistic order: the session label
has already accumulated whatever taint the legitimate work incurred by
the time the attack fires. Each episode runs in a fresh copy of the
suite's environment, and the defended arm resets the session taint label
between episodes.

This design isolates the variable APG actually controls — **the policy
layer's decision on each tool call** — from model behavior. There is no
LLM in the loop, so the numbers are deterministic and free: the
no-defense arm shows what a fully hijacked agent achieves unimpeded, and
the apg arm shows what survives the policy. An end-to-end LLM run (where
attacks succeed only sometimes, and models refuse some tasks on their
own) is the R55 harness's territory.

### Metrics

* **utility** — share of episodes where *every* user-task call executed
  (`EpisodeSummary.task_success`), over all episodes. 100% by
  construction under no defense; under the policy, the loss is the cost
  of gating.
* **armed** — episodes whose script contains at least one attack call on
  a classified external sink (the deny surface of
  `policies/agentdojo.yaml`). This is the ASR denominator.
* **ASR (sink)** — share of *armed* episodes where at least one attack
  sink call executed. The headline attack-success figure.
* **ASR (any-call)** — share of episodes with any attack call where at
  least one executed, whatever the tool. Reported for transparency; see
  below for why it stays non-zero under defense.
* **refusals** — total policy-refused calls across the arm (user and
  attack calls alike).

### Which episodes the ASR excludes, and why

9 of the 35 default injection tasks (1 travel, 8 workspace) have **empty
scripted ground truths**: their goal is exfiltration through the model's
*output text* (e.g. "put the victim's data in your reply"), so a
tool-call replay has nothing to execute and neither arm can measure
them. They are excluded from the ASR denominator ("armed" episodes)
rather than silently deflating both arms' rates.

Slack's `injection_task_3` attacks *via a reader*: its ground truth is a
single `get_webpage` call whose URL exfiltrates data to an
attacker-controlled site. Under the R49b classification `get_webpage` is
an untrusted **reader** (its return carries attacker text), not a sink,
so the call executes in both arms — that is most of why slack's
any-call ASR stays at 60% while its sink ASR is 0%. The chain-level
policy below (R53) closes this channel; the tables above deliberately
keep the R49b classification so the numbers measure the shipped
baseline policy as-is.

## Chain-level policy arm (R53)

[`policies/agentdojo-chain.yaml`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/policies/agentdojo-chain.yaml)
is the baseline policy plus one chain-level rule
(`deny-web-fetch-after-untrusted-read`): deny `get_webpage` once **any
executed call in the session** has returned attacker-authorable content
— a condition on the recorded call history (`chain.any_prior`, R53),
not on the current call's input label. Replayed on slack with a
history-tracking gateway
(`python -m agent_policy_gateway.agentdojo_benchmark slack --policy
policies/agentdojo-chain.yaml`):

```text
suite      arm        episodes  utility  armed ASR(sink) ASR(any-call) refusals
-------------------------------------------------------------------------------
slack      apg-chain       105     4.8%     84      0.0%         40.0%      382
```

Read against the baseline `apg` row above: the any-call ASR drops from
**60% to 40%** — the entire reader-borne channel. `injection_task_3`'s
`get_webpage` attack call, which executed in all 21 of its episodes
under the baseline, is refused in every one of them (refusals 296 →
382), and utility is **unchanged** at 4.8%: no still-succeeding slack
user task fetches a webpage after an untrusted read, so the extra rule
costs nothing here. The remaining 40% is attack calls on plain readers
(`read_channel_messages` etc.) that carry no data out by themselves.
The other three suites don't expose `get_webpage`, so their numbers are
identical to the baseline arm.

One honest caveat: on this scripted replay the same numbers would fall
out of an input-taint rule on `get_webpage` (session taint accumulates
monotonically here). The chain form is deliberately stated over
*history*: it still fires after a declassify grant (R52) strips the
session label, and it can match denied *attempts* — behaviors an input
label cannot express. The integration tests pin the slack figures
above.

## Reproducing

From the repository root, with the `agentdojo` extra installed
(`pip install -e '.[agentdojo]'`):

```bash
python -m agent_policy_gateway.agentdojo_benchmark
```

That replays all four suites, both arms (~1,900 episodes, ≈90 s, no
network or API keys needed) and prints the table above. Options:

```bash
# one suite, machine-readable stats
python -m agent_policy_gateway.agentdojo_benchmark banking --json

# keep every per-episode summary for offline analysis (JSONL, one
# episode per line, deterministic episode ids so arms line up)
python -m agent_policy_gateway.agentdojo_benchmark --episodes-out episodes.jsonl

# a different policy for the defended arm
python -m agent_policy_gateway.agentdojo_benchmark --policy my-policy.yaml
```

Exit codes: **0** success, **2** unknown suite or missing policy file,
**3** `agentdojo` not installed. Per-call decisions additionally land in
the gateway's audit log the same way as any other gated traffic.

The integration tests
(`tests/test_agentdojo_benchmark.py`, skipped unless `agentdojo` is
importable) re-run the banking matrix and assert the banking figures in
the table above, so a change that moves the published numbers fails the
suite.

## Limitations

* Scripted replay measures the **policy layer**, not end-to-end agent
  security: a real LLM sometimes ignores injections on its own, and
  sometimes finds non-ground-truth action sequences. Treat the
  no-defense ASR as "what a fully hijacked agent achieves", an upper
  bound on attacker success that the defense is measured against.
* Success is derived from **call execution**, not environment state or
  AgentDojo's own `utility()`/`security()` checkers (which need model
  output text that scripted replay does not produce). The R49b example's
  environment-state check (`attacker_was_paid`) cross-validates the
  refusal semantics for the banking exfiltration case.
* Output-channel exfiltration (the 9 unarmed injection tasks) is out of
  scope for a tool-call gateway operating alone; measuring it needs the
  R55 long-horizon harness with a model in the loop.

## Relation to the overhead benchmarks

The [overhead page](../benchmarks.md) measures the gateway's per-call
latency cost with `apg-bench`; this page measures its *security value*
on a public benchmark. Both are meant to be tracked across releases.
