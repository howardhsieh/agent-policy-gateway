# TraceSig export

APG is the *prevention* half of a pair whose *detection* half is the
sibling **TraceSig** project: Sigma-style rules matched over agent
tool-call traces. This page freezes the trace format that pairing
consumes — a flat, Sigma-friendly JSONL event stream derived one-to-one
from APG's [audit log](design.md) — and documents the field mapping.

```console
$ apg audit export audit.jsonl --format tracesig -o trace.jsonl
```

`-` reads the audit log from stdin; omitting `--output` writes the
events to stdout. Exit codes mirror the rest of the `apg audit` family:
`0` ok, `2` missing input file or unwritable output path, `3` malformed
audit log line. The mapping itself lives in
`agent_policy_gateway.tracesig_export` (`record_to_event`,
`event_to_record`, `export_events`, `write_tracesig`, `read_tracesig`)
and is pure, so other sinks can reuse it without the CLI.

## Schema versioning

Every event carries two identifying fields:

| Field | Value |
| --- | --- |
| `schema` | Always `apg-audit-trace`. |
| `schema_version` | Currently **1** (an integer). |

The compatibility promise: **within one version, existing fields keep
their name, type, and meaning; new *optional* fields may appear.** Any
change that renames, retypes, or repurposes a field bumps the version.
A TraceSig rule pack should pin the `schema_version` it was written
against and treat an unknown version as a hard error —
`event_to_record` does exactly that.

## Event fields

One JSON object per line, one line per audit record, in log order. Keys
are serialized sorted, so exports diff cleanly.

### Always present

| Field | Type | From the audit record |
| --- | --- | --- |
| `schema` | string | (constant `apg-audit-trace`) |
| `schema_version` | int | (constant, currently 1) |
| `seq` | int | 0-based position of the record in the exported log |
| `ts` | string | `ts` (ISO-8601 UTC; lexicographic order is chronological) |
| `tool` | string | `call.tool_name` |
| `agent` | string \| null | `call.agent_id` |
| `call_id` | string \| null | `call.call_id` |
| `args` | object | `call.args` (nested; Sigma dot-notation reaches into it, e.g. `args.to`) |
| `verdict` | string | `decision.verdict` (`allow` / `deny` / `review` / `redact`) |
| `rule` | string \| null | `decision.rule_id` (`null` = the gateway's default disposition) |
| `reason` | string | `decision.reason` |
| `input_sources` | string[] | `call.input_label.sources` (legacy both-dimension set, sorted) |
| `input_confidentiality` | string[] | `call.input_label.confidentiality` (sorted) |
| `input_integrity` | string[] | `call.input_label.integrity` (sorted) |
| `output_sources` | string[] | `decision.output_label.sources` (sorted) |
| `output_confidentiality` | string[] | `decision.output_label.confidentiality` (sorted) |
| `output_integrity` | string[] | `decision.output_label.integrity` (sorted) |
| `input_untrusted` | bool | *derived*: the input label's effective integrity set is non-empty |
| `input_secret` | bool | *derived*: the input label's effective confidentiality set is non-empty |
| `flagged` | bool | *derived*: `verdict` is `deny` or `review` |

The six label lists are the label's **stored (canonical) dimension
sets**, not the effective per-dimension unions — that is what makes the
mapping lossless. A source in `*_sources` counts in *both* dimensions
(the pre-R51 legacy convention), which the derived booleans already
account for: a rule that wants "any untrusted input" matches
`input_untrusted: true` rather than unioning lists itself.

### Present only when the record carries them

| Field | Type | From the audit record |
| --- | --- | --- |
| `redacted_fields` | string[] | `decision.redacted_fields` (R17 redact action) |
| `declassified_by` | string[] | `decision.declassified_by` (R52 declassify grant ids that fired) |
| `input_provenance` | object[] | `call.input_provenance` — `{source, tool, call_id}` per hop (R19 provenance, `tool` = `tool_name`) |
| `output_provenance` | object[] | `decision.output_provenance`, same entry shape |
| `arg_labels` | object | `call.arg_labels` (R57 per-value labels): `{argname: {sources, confidentiality, integrity}}`, each list sorted |
| `prev` | string | the record's R27 hash-chain digest (64 hex chars; genesis is all zeros) |

Absence means the audit record did not carry the field, so a legacy log
exports without invented fields.

### Derived fields

`seq`, `input_untrusted`, `input_secret`, and `flagged` are computed
*from* the other fields for rule-authoring convenience. They are ignored
(recomputed, never trusted) when an event is mapped back to an audit
record.

## Losslessness

`event_to_record` inverts `record_to_event` **exactly** — the round-trip
test in `tests/test_tracesig_export.py` pins
`event_to_record(record_to_event(r, seq)) == r` across a battery that
includes legacy-minimal records, provenance chains, per-value labels,
declassification, and hash-chained `prev` digests. TraceSig itself only
reads events; `read_tracesig` exists so tests and tooling can prove the
export lost nothing.

## Example traces

Three committed sessions under
[`examples/traces/`](https://github.com/howardhsieh/agent-policy-gateway/tree/main/examples/traces)
— a benign session, a denied injection, and a laundering chain — each
as the raw audit log plus its TraceSig export, generated through the
real gateway with a deterministic clock. Regenerate them with:

```console
$ python -m examples.traces
```

The command rewrites all six files and exits non-zero if any narrative
invariant stops holding; CI compares the committed bytes against an
in-process regeneration, so the fixtures cannot drift silently.
