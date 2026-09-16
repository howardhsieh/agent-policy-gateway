# APG / Progent / Fides comparison demo (R56 + R57)

Replays one shared long-horizon scenario family under **seven arms** — no
defense, real imported Progent symbolic rules, a Fides-style dual-label
IFC policy, APG's input-taint / chain / selective-chain policies, and the
R57 per-value taint policy — and prints the per-variant utility and
compromise matrices.

```bash
python -m examples.comparison
```

prints the two matrices and a pass/fail line per R56/R57 invariant,
exiting non-zero if any fails (so it doubles as a CI check).

## What each scenario does

Every scenario is one persistent session, as in the R55 demo, but the
family carries the observables the paradigms need to come apart: every
sink call has a `recipient` argument (trusted namespace / novel /
attacker-controlled) **and a `payload` argument** (the agent's own clean
text on benign flows; the read value — direct, laundered through
`sanitize`, or the secret itself — on attacks), and a confidential
`read_secret` source joins the untrusted reader.

- **Benign variants** (utility): `clean`, `novel` (clean session,
  recipient outside the allowlist), `direct` and `launder` (after an
  untrusted read, without/with a vetted sanitize), `secret` (a
  legitimate flow that touched secret data).
- **Armed variants** (compromise): `overt-direct` / `overt-launder`
  (attacker recipient), `covert-direct` / `covert-launder` (trusted
  recipient — exfiltration through a permitted channel), `exfil`
  (secret data to a permitted recipient).

## The seven arms

| arm | policy | paradigm |
|-----|--------|----------|
| `no-defense` | — | bare runtime |
| `progent` | [`policies/comparison-progent.json`](../../policies/comparison-progent.json) via the R54 importer | stateless symbolic argument rules |
| `fides` | [`policies/comparison-fides.yaml`](../../policies/comparison-fides.yaml) | dual-label IFC (R51) with integrity-only endorse |
| `apg-input-taint` | [`policies/stateful-input-taint.yaml`](../../policies/stateful-input-taint.yaml) | current session label (R49a/R52) |
| `apg-chain` | [`policies/stateful-chain.yaml`](../../policies/stateful-chain.yaml) | session call history (R53) |
| `apg-chain-selective` | [`policies/comparison-chain-selective.yaml`](../../policies/comparison-chain-selective.yaml) | chain history, selective on the recipient argument (R54 `arg_matches`) |
| `apg-value-taint` | [`policies/comparison-value-taint.yaml`](../../policies/comparison-value-taint.yaml) | per-value labels on the payload argument (R57 `arg_taint` over the value ledger) |

## The findings

Among the session-scoped arms no arm dominates — the frontier is the
R56 result — and the R57 arm shows what the finer observable buys:

1. laundering flips the stateless label arms while both chain arms hold;
2. the selective chain arm recovers **full** utility (novel recipient
   included) at Progent's compromise profile — session state strictly
   dominates the equivalent stateless allowlist;
3. covert attacks pass every *session-scoped* arm that allows trusted
   recipients: at that scope they are observationally identical to the
   legitimate flow. The per-value arm separates them — the covert
   payload value carries the read's label, the benign payload never did;
4. among session-scoped arms only the Fides-style one stops
   exfiltration, paying for it on the benign secret flow; the per-value
   arm holds `exfil` at 0% *and* passes the benign `secret` flow,
   dissolving that one-for-one trade;
5. the per-value arm's 100% utility / 0% compromise holds **only on
   derivations the runtime mediates** — exact-match value propagation is
   the PoC boundary; a paraphrased exfiltration re-opens the residual
   (R59 territory).

The full write-up, with the numbers, is in
[`docs/benchmarks/comparison.md`](../../docs/benchmarks/comparison.md);
the benchmark is `agent_policy_gateway.comparison_benchmark`.
