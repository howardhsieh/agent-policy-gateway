# APG / Progent / Fides comparison demo (R56)

Replays one shared long-horizon scenario family under **six arms** — no
defense, real imported Progent symbolic rules, a Fides-style dual-label
IFC policy, and APG's input-taint / chain / selective-chain policies —
and prints the per-variant utility and compromise matrices.

```bash
python -m examples.comparison
```

prints the two matrices and a pass/fail line per R56 invariant, exiting
non-zero if any fails (so it doubles as a CI check).

## What each scenario does

Every scenario is one persistent session, as in the R55 demo, but the
family carries the observables the paradigms need to come apart: every
sink call has a `recipient` argument (trusted namespace / novel /
attacker-controlled), and a confidential `read_secret` source joins the
untrusted reader.

- **Benign variants** (utility): `clean`, `novel` (clean session,
  recipient outside the allowlist), `direct` and `launder` (after an
  untrusted read, without/with a vetted sanitize), `secret` (a
  legitimate flow that touched secret data).
- **Armed variants** (compromise): `overt-direct` / `overt-launder`
  (attacker recipient), `covert-direct` / `covert-launder` (trusted
  recipient — exfiltration through a permitted channel), `exfil`
  (secret data to a permitted recipient).

## The six arms

| arm | policy | paradigm |
|-----|--------|----------|
| `no-defense` | — | bare runtime |
| `progent` | [`policies/comparison-progent.json`](../../policies/comparison-progent.json) via the R54 importer | stateless symbolic argument rules |
| `fides` | [`policies/comparison-fides.yaml`](../../policies/comparison-fides.yaml) | dual-label IFC (R51) with integrity-only endorse |
| `apg-input-taint` | [`policies/stateful-input-taint.yaml`](../../policies/stateful-input-taint.yaml) | current session label (R49a/R52) |
| `apg-chain` | [`policies/stateful-chain.yaml`](../../policies/stateful-chain.yaml) | session call history (R53) |
| `apg-chain-selective` | [`policies/comparison-chain-selective.yaml`](../../policies/comparison-chain-selective.yaml) | chain history, selective on the recipient argument (R54 `arg_matches`) |

## The findings

No arm dominates — the point of the measurement is the frontier:

1. laundering flips the stateless label arms while both chain arms hold;
2. the selective chain arm recovers **full** utility (novel recipient
   included) at Progent's compromise profile — session state strictly
   dominates the equivalent stateless allowlist;
3. covert attacks pass every arm that allows trusted recipients: they
   are observationally identical to the legitimate flow;
4. only the Fides-style arm stops exfiltration, and it pays for it on
   the benign secret flow.

The full write-up, with the numbers, is in
[`docs/benchmarks/comparison.md`](../../docs/benchmarks/comparison.md);
the benchmark is `agent_policy_gateway.comparison_benchmark`.
