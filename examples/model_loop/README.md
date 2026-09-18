# Model-in-the-loop benchmark demo (R59)

Puts an **agent driver** — instead of a fixed script — in the loop of a
persistent AgentDojo banking session: the driver sees the task prompt,
every tool result (attacker injections included), and every policy
refusal, and decides the next call itself.

```bash
pip install -e ".[agentdojo]"
python -m examples.model_loop
```

replays the committed driver fixtures (`fixtures/*.jsonl`) — key-free
and network-free — prints the benchmark table, and checks the R59
invariants, exiting non-zero if any fails (so it doubles as a CI check).

## The rows

| arm | reaction | meaning |
|-----|----------|---------|
| `no-defense` | — | bare runtime; the injected instructions steer the agent |
| `apg-input-taint` | `retry` | on a refusal, re-issue the call verbatim (up to 2×), then move on |
| `apg-input-taint` | `skip` | on a refusal, drop that call and continue the task |
| `apg-input-taint` | `abort` | on a refusal, abandon the rest of the turn and apologize |

Utility and attack success are judged by AgentDojo's own per-task
`utility` / `security` checks over environment snapshots. The extra
columns — refusals, verbatim retries, driver steps, tokens, wall-clock —
are what a scripted replay cannot show: how the agent *behaves* under
policy pressure and what that behavior costs.

## Provenance of the fixtures (read this before citing numbers)

The committed fixtures are recorded from the **deterministic simulated
agent** (`SimulatedAgentDriver`): faithful to the user task's ground
truth, hijacked the first time it sees the `<INFORMATION>` injection
marker in a tool result, reacting to refusals by the row's strategy.
It is **not a real LLM**, and its token counts are size-based estimates
(flagged `estimated`, rendered with `~`). Recording a real model uses
the identical plumbing and replaces the fixtures:

```bash
export ANTHROPIC_API_KEY=...
python -m agent_policy_gateway.model_benchmark \
    --mode record --provider anthropic --model claude-haiku-4-5-20251001
```

after which `--mode replay` (and this demo) replays the real model's
session deterministically, with provider-measured token counts. See
`docs/benchmarks/model-loop.md` for the full write-up.
