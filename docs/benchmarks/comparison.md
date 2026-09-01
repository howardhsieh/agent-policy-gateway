# APG / Progent / Fides comparison (R56)

The R55 long-horizon benchmark ([stateful.md](stateful.md)) quantified a
tension *inside* APG: an input-taint policy is laundered by a legitimate
mid-session declassify, a chain-history policy holds — at a utility cost.
R56 widens that measurement into a comparison across the three policy
**paradigms** the project sits between, on one shared scenario family:

```bash
python -m agent_policy_gateway.comparison_benchmark
```

Deterministic, no `agentdojo` package, no API keys; every number below is
pinned by `tests/test_comparison_benchmark.py` and re-asserted by
`python -m examples.comparison`.

## The arms

| arm | policy | paradigm |
|-----|--------|----------|
| `no-defense` | — | bare runtime |
| `progent` | [`policies/comparison-progent.json`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/policies/comparison-progent.json) | **Progent-style stateless symbolic rules**: per-call conditions on tool name + arguments, no session state. The file is real Progent-format JSON translated at run time through the R54 importer, so the arm measures the actual import pipeline. |
| `fides` | [`policies/comparison-fides.yaml`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/policies/comparison-fides.yaml) | **Fides-style dual-label IFC** (R51): integrity rules (unendorsed untrusted content cannot reach a sink) + confidentiality rules (secret data cannot reach an external sink), with an integrity-only *endorse* grant for the vetted sanitizer. |
| `apg-input-taint` | [`policies/stateful-input-taint.yaml`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/policies/stateful-input-taint.yaml) | current session label (R49a) + declassify grant (R52) — the R55 arm, unchanged |
| `apg-chain` | [`policies/stateful-chain.yaml`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/policies/stateful-chain.yaml) | session call history (R53) — the R55 arm, unchanged |
| `apg-chain-selective` | [`policies/comparison-chain-selective.yaml`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/policies/comparison-chain-selective.yaml) | **the R56 refinement**: trusted-namespace recipients (`arg_matches`, R54) are allowed first-match, then the chain rule denies what remains after an untrusted read |

Every defended arm wraps the same runtime behind the same
history-tracking gateway with the same taint specs; the arms differ in
nothing but their policy.

## The scenario family

90 persistent multi-turn scenarios (3 sinks × horizons 1–3), replayed by
the R55 harness with state carried across turns. Beyond the R55 family,
sink calls carry a `recipient` argument — `trusted:*` (the namespace a
policy may allowlist), `new:bob` (legitimate but novel), or
`evil:attacker` — and a confidential source `read_secret` (taints only
the confidentiality dimension) joins the untrusted reader.

**Benign variants** (utility): `clean` (no read), `novel` (clean session,
recipient outside the trusted namespace), `direct` / `launder` (after an
untrusted read, without / with a vetted `sanitize`), `secret` (a
legitimate flow that touched secret data). **Armed variants**
(compromise): `overt-direct` / `overt-launder` (attacker recipient),
`covert-direct` / `covert-launder` (trusted recipient — exfiltration
through a permitted channel), `exfil` (secret data to a permitted
recipient).

## Results

```text
compromise per attack variant
arm                   utility   comp.   ov-dir  ov-lau  cv-dir  cv-lau   exfil
------------------------------------------------------------------------------
no-defense             100.0%  100.0%   100.0%  100.0%  100.0%  100.0%  100.0%
progent                 80.0%   60.0%     0.0%    0.0%  100.0%  100.0%  100.0%
fides                   60.0%   40.0%     0.0%  100.0%    0.0%  100.0%    0.0%
apg-input-taint         80.0%   60.0%     0.0%  100.0%    0.0%  100.0%  100.0%
apg-chain               60.0%   20.0%     0.0%    0.0%    0.0%    0.0%  100.0%
apg-chain-selective    100.0%   60.0%     0.0%    0.0%  100.0%  100.0%  100.0%

utility per benign variant
arm                   utility    clean   novel  direct launder  secret
----------------------------------------------------------------------
no-defense             100.0%   100.0%  100.0%  100.0%  100.0%  100.0%
progent                 80.0%   100.0%    0.0%  100.0%  100.0%  100.0%
fides                   60.0%   100.0%  100.0%    0.0%  100.0%    0.0%
apg-input-taint         80.0%   100.0%  100.0%    0.0%  100.0%  100.0%
apg-chain               60.0%   100.0%  100.0%    0.0%    0.0%  100.0%
apg-chain-selective    100.0%   100.0%  100.0%  100.0%  100.0%  100.0%
```

## Findings

1. **Laundering separates label state from history state — across
   paradigms.** The R55 finding generalizes: the two arms that decide on
   a *current label* (`fides`, `apg-input-taint`) stop every overt direct
   attack but are flipped to 100% by the mid-session endorse/declassify;
   the two arms that decide on the *call history* hold both overt
   variants at 0%. Which paradigm a policy belongs to matters less than
   whether the state it reads can be laundered.

2. **Session state strictly dominates the equivalent stateless
   allowlist.** `progent` and `apg-chain-selective` guard the same
   trusted-recipient namespace and end with the *same compromise profile*
   (60.0%, identical per variant) — but the stateless allowlist must
   refuse the novel recipient even in a perfectly clean session (`novel`
   utility 0%, overall 80%), while the selective chain arm only tightens
   after the session has actually read untrusted content (100% utility).
   Conversely the chain condition alone (`apg-chain`, 60% utility) pays
   for its robustness on every post-read flow; adding the argument
   condition recovers all of it without reopening a single overt attack.

3. **Covert attacks are the residual no call-level policy can close.** A
   covert attack — the injected sink call using a *trusted* recipient —
   is observationally identical to the legitimate laundered flow: same
   tool, same arguments, same session state. Accordingly, in every arm
   `covert-launder` compromise equals `launder` benign utility (both
   allowed or both refused; the tests pin this equivalence). The choice
   is only *where* to pay: `apg-chain` refuses both (secure, 60%
   utility), the selective arms allow both (100% utility, 60%
   compromise). Closing it needs finer observables than tool + args +
   session state — per-value labels or content inspection.

4. **Only the confidentiality-aware arm stops exfiltration — and it pays
   for it.** `fides` is the sole arm holding `exfil` at 0% (secret data
   to a permitted recipient), because only it carries a
   confidentiality-dimension rule; every integrity-only arm passes it.
   The price appears in the same column: the benign `secret` flow is
   refused too (fides utility 60%). At session granularity,
   confidentiality coverage and secret-touching utility trade one for
   one — per-value labels (Fides proper) are the way out, and the R51
   dimension machinery is where they would land in APG.

5. **No arm dominates: the frontier is the result.** `apg-chain` is the
   most robust (20% compromise, 60% utility), `apg-chain-selective` the
   most useful among defended arms (100% utility, 60% compromise),
   `fides` the only exfiltration cover (40% compromise, 60% utility).
   The three sit on a Pareto frontier a deployment picks from —
   `no-defense`, `progent` and `apg-input-taint` are dominated on it.

## What is measured (and what is not)

Like R50/R55 this is a **deterministic scripted replay** — no model
decides whether to follow an injection, and "compromise" means an
injected call executed, not that harm resulted. The `fides` arm is a
*session-granular* rendering of Fides' label discipline in APG's own
policy language, not the Fides runtime: Fides proper labels individual
values and plans over them, which would place it between the label and
history arms here (a per-value endorse would not launder unrelated
data). The `progent` arm imports real Progent-format rules through R54
but represents the paper's symbolic subset, not its LLM-generated
dynamic policies. Numbers are exact rates over the 90-scenario family,
not samples.
