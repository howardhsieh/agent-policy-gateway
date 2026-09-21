# Example audit traces for TraceSig (R62)

APG is the *prevention* half of a pair whose *detection* half is the
sibling **TraceSig** project (Sigma-style rules over agent tool-call
traces). This directory holds three small sessions TraceSig can vendor
as test fixtures, each committed twice:

| Scenario | Audit log | TraceSig export |
| --- | --- | --- |
| Benign session | `benign-session.audit.jsonl` | `benign-session.tracesig.jsonl` |
| Denied injection | `denied-injection.audit.jsonl` | `denied-injection.tracesig.jsonl` |
| Laundering chain | `laundering-chain.audit.jsonl` | `laundering-chain.tracesig.jsonl` |

The narratives, in one line each:

* **Benign session** — two internal KB lookups and a clean outbound
  email under `policies/default.yaml`; every call allowed. The baseline
  a detection rule must stay silent on.
* **Denied injection** — a `web_fetch` taints the session (`web`
  source, provenance pinned to its `call_id`), and the exfiltration
  email to `attacker@evil.example` is denied by `deny-web-to-email`.
* **Laundering chain** — under `policies/stateful-chain.yaml` with
  session history tracking and a hash-chained audit writer: an
  untrusted `read_inbox`, a vetted `sanitize` whose declassify grant
  strips the label, then a `send_money` arriving with a *clean* label —
  still denied by the R53 chain rule. The trace a label-only detection
  rule cannot flag and a history-aware one can.

Everything is generated through the real `Gateway` + `JsonlAuditWriter`
with a deterministic clock, so regeneration is byte-for-byte
reproducible:

```console
$ python -m examples.traces
```

The exporter and the frozen `apg-audit-trace` schema (version, field
mapping) are documented in [`docs/tracesig-export.md`](../../docs/tracesig-export.md);
the CLI entry point is `apg audit export LOG --format tracesig`.
`tests/test_tracesig_export.py` regenerates these fixtures in-process
and fails CI if the committed bytes drift.
