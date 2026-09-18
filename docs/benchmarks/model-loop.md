# Model-in-the-loop long-horizon eval (R59)

Every benchmark before this page replays a *script*: the ground-truth
calls a faithful agent would make, plus the calls a fully hijacked agent
would attempt (R49c/R50), composed into persistent multi-turn sessions
(R55/R56/R57). Scripts pin down the *policy's* behavior exactly, but they
cannot answer the question R56 finding 6 left open: what does the **agent**
do when a policy starts refusing its calls — retry, reroute, give up? —
and what does that behavior cost in utility, attack success, tokens, and
wall-clock?

R59 closes the loop. A **driver** — a live LLM, a recorded replay of one,
or the deterministic simulated agent described below — sits where the
script used to be: it sees the task prompt, every tool result (attacker
injections included), and every policy refusal verbatim, and decides the
next call itself.

## Method

**Harness.** `agent_policy_gateway.model_loop.run_model_scenario` keeps
the R55 persistent-session shape: several user-task turns run as *one*
conversation over *one* runtime, with the session taint reset once at the
start and never between turns. Because the model decides the calls, the
scripted `user`/`attack` labels no longer exist; outcome semantics move to
the environment, AgentDojo-style — per-turn **utility** and
scenario-level **security** are judged by AgentDojo's own checks over
deep-copied environment snapshots. Refusals surface to the driver as the
adapter's `PolicyDenied: refused by rule '…'` tool errors (R49a), so the
agent gets exactly the feedback a real framework would give it.

**Driver seam.** `agent_policy_gateway.model_driver` defines the
provider-neutral protocol (`next_step(messages, tools) → ModelStep`), a
live `AnthropicDriver`, and the record/replay pair: `RecordingDriver`
persists every (request, step) to JSONL keyed by a SHA-256 digest of the
exact request; `ReplayDriver` plays the file back, verifying each digest —
CI re-runs the *identical* session with no key and no network, and any
drift in prompts, tool schemas, or harness plumbing fails loudly instead
of silently reporting stale numbers.

**Slice.** Three persistent banking-suite sessions
(`model_benchmark.DEFAULT_SLICE`), each reading at least one injection
vector so the attack is always presented, with the "important
instructions" template carrying the injection task's goal:

| scenario | turns | injection task |
|----------|-------|----------------|
| `banking:s1:spending+dinner` | `user_task_1` (query) → `user_task_3` (send_money) | `injection_task_0` |
| `banking:s2:pay-bill` | `user_task_0` (read bill → send_money) | `injection_task_5` (max exfil) |
| `banking:s3:address+password` | `user_task_13` (file → update_user_info) → `user_task_14` (update_password) | `injection_task_8` (two-call exfil) |

**Arms.** `no-defense` (bare AgentDojo `FunctionsRuntime`) vs
`apg-input-taint` (`policies/agentdojo.yaml` via `gate_suite`: sinks deny
while the session label carries `agentdojo:untrusted`). The defended arm
runs under three refusal-reaction strategies: **retry** (re-issue the
refused call verbatim, up to 2×), **skip** (drop it, continue), **abort**
(abandon the rest of the turn and apologize).

## Provenance — read before citing

!!! warning "The committed fixtures are from a simulated agent, not a real LLM"
    This environment has no LLM API key, so the committed fixtures under
    `examples/model_loop/fixtures/` are recorded from
    `SimulatedAgentDriver`: a deterministic, LLM-free stand-in that
    follows the user task's ground truth, adopts the injection task's
    calls the first time it sees the `<INFORMATION>` marker in a tool
    result (i.e. it models the *fully hijacked* agent, the same
    pessimistic assumption as R49c), and reacts to refusals by the row's
    strategy. Its token counts are size-based estimates, flagged
    `estimated` end-to-end and rendered with `~`. The numbers below are
    therefore **harness-validated upper/lower bounds under a known agent
    model**, not measurements of any particular LLM. Recording a real
    model replaces the fixtures through the identical plumbing:

    ```bash
    export ANTHROPIC_API_KEY=...
    python -m agent_policy_gateway.model_benchmark \
        --mode record --provider anthropic --model claude-haiku-4-5-20251001
    ```

## Results

Reproduce with (deterministic; the first two need the `agentdojo` extra,
neither needs a key):

```bash
python -m agent_policy_gateway.model_benchmark            # simulated, live
python -m agent_policy_gateway.model_benchmark --mode replay   # from fixtures
python -m examples.model_loop                             # replay + invariants
```

| arm | reaction | utility | ASR | refusals | verbatim retries | driver steps | tokens (est.) |
|-----|----------|--------:|----:|---------:|-----------------:|-------------:|--------------:|
| no-defense | — | **100.0%** | **100.0%** | 0 | 0 | 18 | ~10,213 |
| apg-input-taint | retry | 20.0% | **0.0%** | 21 | 15 | 32 | ~26,959 |
| apg-input-taint | skip | 20.0% | **0.0%** | 7 | 1 | 18 | ~10,196 |
| apg-input-taint | abort | **0.0%** | **0.0%** | 5 | 0 | 16 | ~8,857 |

Utility is over 5 judged turns, ASR over 3 scenarios;
`tests/test_model_benchmark.py` pins every number, and
`python -m examples.model_loop` asserts the findings below. Wall-clock is
reported per row by the tool (sub-hundredth seconds here — it becomes a
meaningful cost column only with a live provider; per-step provider
latency then dominates).

## Findings

1. **The input-taint policy transfers to the model-in-the-loop setting.**
   ASR closes 100% → 0% under *every* reaction strategy: whether the
   hijacked agent retries, skips, or aborts, the sink stays shut because
   the decision reads the session label, not the agent's persistence.

2. **Persistent sessions concentrate the utility cost on sink turns.**
   Utility falls 100% → 20%: every slice scenario reads an injection
   vector before its sinks fire, so every sink turn is tainted by the
   time it acts. The surviving 20% is the query turn (`user_task_1`),
   whose answer needs no sink — the model-in-the-loop analogue of the
   R55 observation that a persistent session pays the taint cost for the
   whole horizon.

3. **Reaction strategy changes cost, not outcomes — except abort, which
   loses utility the policy never blocked.** Retry vs skip: identical
   utility and ASR, but 3× the refusals (21 vs 7) and ~2.6× the tokens
   (~26,959 vs ~10,196) — under a deterministic policy, retrying a
   refusal is pure spend. Abort is the cheapest in tokens (~8,857) but
   drops to 0% utility: when the refusal makes the agent abandon the
   turn, the apology replaces the final answer, so even the query turn —
   which the policy never touched — fails. Over-compliance with a
   refusal is its own failure mode, and the harness now measures it.

4. **The `retried_refusals` observable works as designed** (a call
   re-issued to the same function immediately after a refusal): 15 of the
   retry arm's 21 refusals are verbatim retries; the lone "retry" in the
   skip arm is the documented upper-bound artifact (an attack `send_money`
   refusal followed by the plan's own `send_money`).

## Caveats and next steps

* The simulated agent is *maximally* hijackable (it always follows the
  injection) and *minimally* creative (it never reroutes through an
  unlisted tool). A real model will land somewhere else on both axes —
  that is exactly what the record mode is for; the fixtures and every
  number above regenerate from one command once a key is available.
* Utility judgments are AgentDojo's own and are not all strict: some
  check the final answer text, some the environment. The slice was
  chosen so every sink-turn check requires a *new* transaction/mutation
  (`user_task_5` was rejected for being satisfiable by pre-existing
  state).
* Token counts for the simulated driver are size-based estimates
  (`chars/4`), useful only for *relative* comparison between rows
  (retry ≈ 2.6× skip); provider-measured counts arrive with real
  recordings, via the same `Usage(estimated=False)` path the
  `AnthropicDriver` already populates.
