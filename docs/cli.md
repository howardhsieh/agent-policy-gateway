# CLI reference

`agent-policy-gateway` installs three console scripts. This page is the complete
reference for the main one, **`apg`**; a test walks the argparse parser and fails
the build if any subcommand or flag documented here drifts from the code.

| Script | Purpose |
| --- | --- |
| `apg` | Inspect and validate policies; summarize audit logs. |
| `apg-replay` | Replay a JSONL audit log as a timeline, or verify its hash chain. |
| `apg-bench` | Micro-benchmark the decision path. See [Benchmarks](benchmarks.md). |

```console
$ pip install -e ".[dev]"
$ apg --help
```

Every command accepts `--help`.

---

## Exit codes

`apg` uses a single exit-code vocabulary across subcommands, so CI can branch on the
number without parsing output.

| Code | Meaning | Emitted by |
| --- | --- | --- |
| `0` | Success — the command ran and found nothing that should fail a build. | every subcommand |
| `1` | Invalid policy: the file parsed but violates the schema (message is line-located). | `policy validate`, `policy explain`, `policy diff`, `policy lint` |
| `2` | Missing file, or an unusable flag combination (`--csv-section` without `--csv`; `-` mixed with paths). | every subcommand |
| `3` | Findings that should fail a build: lint findings, or a malformed audit-log line. | `policy lint`, `audit stats` |
| `4` | Broken audit hash chain. Not emitted by `apg`; reserved here because it is shared with `apg-replay --verify`. | `apg-replay --verify` |
| `5` | CI gate tripped: the deny+review share is **over** `--fail-over`. | `audit stats` |
| `6` | CI gate tripped: the allow share is **under** `--fail-under`. | `audit stats` |

Argparse itself exits `2` on a usage error (unknown flag, missing positional), which
lines up with the "missing/unusable input" meaning above.

When both gates trip, `--fail-over` wins and the exit code is `5`.

---

## `apg policy`

Inspect and validate policy files.

### `apg policy validate`

Parse and validate a policy YAML against the schema.

```console
$ apg policy validate policies/example.yaml
OK: policy 'example' is valid (4 rule(s))
```

| Argument | Description |
| --- | --- |
| `file` | Path to the policy YAML file. |

Exits `0` if valid, `1` with a line-located message if malformed, `2` if the file is
missing.

### `apg policy explain`

Build a hypothetical tool call and print the first-match trace through the policy's
rules — which rules were skipped and why, and which one finally decided.

```console
$ apg policy explain policies/example.yaml \
    --tool send_email --identity mailer --taint web,pii --resource 'mail://*'
```

| Flag | Argument | Default | Description |
| --- | --- | --- | --- |
| `file` (positional) | path | — | Path to the policy YAML file. |
| `--tool` | `NAME` | *required* | Tool name of the hypothetical call (e.g. `send_email`). |
| `--identity` | `ID` | none | Agent identity making the call, matched against the rule identity. |
| `--taint` | `a,b,c` | none | Comma-separated taint sources on the call input (e.g. `web,pii`). |
| `--resource` | `R` | none | Target resource of the call, matched against rule resource globs. |
| `--arg` | `KEY=VALUE` | none | Argument on the hypothetical call. Repeatable. `true`/`false` become bools, decimal integers become ints, everything else stays a string. |

Exits `0` on success, `1` if the policy is malformed, `2` if the file is missing.

### `apg policy diff`

Compare two policies **by the decisions they produce**, not by their text. A matrix of
synthetic calls is derived from both policies' rule selectors, evaluated against each
policy, and every scenario whose first-match decision changed is reported. Passing any
of `--tool`/`--identity`/`--taint`/`--resource`/`--arg` switches to single-scenario mode.

```console
$ apg policy diff policies/old.yaml policies/new.yaml
```

| Flag | Argument | Default | Description |
| --- | --- | --- | --- |
| `old` (positional) | path | — | Path to the old policy YAML file. |
| `new` (positional) | path | — | Path to the new policy YAML file. |
| `--tool` | `NAME` | none | Restrict the diff to a single scenario with this tool name. |
| `--identity` | `ID` | none | Agent identity for the single-scenario diff. |
| `--taint` | `a,b,c` | none | Comma-separated taint sources for the single-scenario diff. |
| `--resource` | `R` | none | Target resource for the single-scenario diff. |
| `--arg` | `KEY=VALUE` | none | Argument on the single-scenario call. Repeatable; same coercion as `policy explain`. Switches diff into single-scenario mode. |

Exits `0` whether or not decisions changed (a diff is informational), `1` if a policy is
malformed, `2` if a file is missing.

### `apg policy lint`

Static quality checks: rules that can never match (a self-contradictory taint clause) and
rules shadowed by an earlier, at-least-as-general rule.

```console
$ apg policy lint policies/example.yaml
OK: no lint findings in policy 'example' (4 rule(s))
```

| Argument | Description |
| --- | --- |
| `file` | Path to the policy YAML file. |

Exits `0` when clean, **`3` when findings were reported** (so CI fails), `2` if the file
is missing, `1` if the policy is malformed.

---

## `apg audit`

Inspect JSONL audit logs.

### `apg audit stats`

One-screen summary of one or more audit logs: record count, first/last timestamp, counts
by verdict, the deny+review share, and the top rules, tools and agents by hit count.
Several logs are summarized as the concatenation of their records in argument order.

```console
$ apg audit stats audit.jsonl --top 10
$ cat audit.jsonl | apg audit stats - --json
$ apg audit stats audit.jsonl --verdict deny --since 2026-08-01 --fail-over 5
```

#### Input

| Argument | Description |
| --- | --- |
| `log` (positional, one or more) | Path(s) to JSONL audit log file(s). `-` reads a single log from stdin and may not be combined with file paths. |

#### Ranked-list size

| Flag | Argument | Default | Description |
| --- | --- | --- | --- |
| `--top` | `N` | `5` | Show the top N rules, tools and agents by hit count. |
| `--top-rules` | `N` | `--top` | Cap the top-rules list at N, overriding `--top` for this list only. |
| `--top-tools` | `N` | `--top` | Cap the top-tools list at N, overriding `--top` for this list only. |
| `--top-agents` | `N` | `--top` | Cap the top-agents list at N, overriding `--top` for this list only. |

#### Output format

`--json`, `--csv` and `--count-only` are mutually exclusive; omitting all three gives the
human-readable text summary.

| Flag | Argument | Default | Description |
| --- | --- | --- | --- |
| `--json` | — | off | Emit the summary as machine-readable JSON. |
| `--csv` | — | off | Emit a CSV table for piping into a spreadsheet. |
| `--count-only` | — | off | Print only the number of records left after filtering (a single integer line), for shell pipelines. Still runs the CI gates. |
| `--csv-section` | `verdicts\|rules\|tools\|agents` | `verdicts` | Which breakdown `--csv` emits: the per-verdict table (default) or a ranked `<name>,count,pct` table honoring `--top` and the per-list caps. Requires `--csv`; using it without `--csv` exits `2`. |

#### Filters

Filters compose; every figure in the summary reflects the filtered subset. Include
filters run first, then the `--exclude-*` filters.

| Flag | Argument | Default | Description |
| --- | --- | --- | --- |
| `--verdict` | `{allow,deny,review,redact}` | all | Only summarize records with the given verdict. Repeatable to include several. |
| `--since` | `TS` | none | Only records at or after this ISO-8601 timestamp (inclusive; an ISO prefix like `2026-06-13` works). |
| `--until` | `TS` | none | Only records at or before this ISO-8601 timestamp (inclusive; an ISO prefix works). |
| `--tool` | `GLOB` | all | Only records whose tool name matches this fnmatch glob (e.g. `send_*`). Repeatable — patterns union. |
| `--agent` | `GLOB` | all | Only records whose agent id matches this fnmatch glob. Repeatable. Select unattributed traffic with the sentinel `(unattributed - no agent_id)`. |
| `--rule` | `GLOB` | all | Only records whose matched rule id matches this fnmatch glob (e.g. `deny-*`). Repeatable. Select default/unruled traffic with the sentinel `(default - no rule)`. |
| `--exclude-tool` | `GLOB` | none | Drop records whose tool name matches this glob. Repeatable; applied after `--tool`. |
| `--exclude-agent` | `GLOB` | none | Drop records whose agent id matches this glob. Repeatable; applied after `--agent`. |
| `--exclude-rule` | `GLOB` | none | Drop records whose matched rule id matches this glob. Repeatable; applied after `--rule`. |

#### CI gates

Both gates print the summary first and only then set the exit code, so a failing CI job
still shows the numbers that failed it. Both are off by default.

| Flag | Argument | Default | Description |
| --- | --- | --- | --- |
| `--fail-over` | `PCT` | off | Exit `5` when the combined deny+review share exceeds PCT percent. Catches an over-permissive agent. An empty log is never over threshold. |
| `--fail-under` | `PCT` | off | Exit `6` when the allow share falls below PCT percent. Catches an over-restrictive policy. An empty log has a `0.0` allow share and so trips any positive threshold. |

Combined, they bracket an acceptable band. If both trip, `--fail-over` wins (exit `5`).

Exits `0` on success, `2` if a file is missing (or `-` is mixed with paths), `3` if a log
line is malformed, `5`/`6` per the gates above.

---

## Sibling scripts

### `apg-replay`

```console
$ apg-replay audit.jsonl --limit 20 --verdict deny
$ apg-replay audit.jsonl --verify
```

Prints the log as a human-readable timeline. `--limit N` truncates, `--verdict` filters.
`--verify` instead checks the tamper-evident hash chain and exits `0` if intact or `4` if
a link is broken (the first broken line number goes to stderr).

### `apg-bench`

Micro-benchmarks the decision path; see [Benchmarks](benchmarks.md) for the methodology
and the tracked numbers.
