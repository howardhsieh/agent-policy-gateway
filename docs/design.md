# Design

## Position in the agent stack

```
+----------------+         +-------------------+         +-----------+
|   LLM / Agent  |  --->   |  Policy Gateway   |  --->   |   Tool    |
+----------------+         +-------------------+         +-----------+
                            ^         ^      ^
                            |         |      |
                       policies   audit log  taint store
```

The gateway is a **reference monitor**: every tool call passes through it, every decision
is logged, and policies cannot be bypassed by the LLM.

## Information flow control (IFC)

We borrow the classic lattice model from OS-level IFC: each piece of data carries a label,
labels form a lattice under join (`∨`) and order (`⊑`), and the gateway enforces
non-interference rules at sinks.

Concretely, every tool output is tagged with a set of *source* labels — strings like
`web`, `user_upload`, `crm.contact.email`. When tool A's output is passed as an argument
to tool B, B's effective input label is the join of all argument labels. Policies on B can
then refuse, require human review, or downgrade based on the input label.

This is a coarse approximation of full IFC — we don't track field-level taint inside
JSON outputs (yet) — but it's enough to catch the dominant exfiltration patterns:

- *Indirect prompt injection*: a malicious web page tells the agent to email the user's
  contacts. Without IFC, the email send looks fine. With IFC, the send's `to` and `body`
  carry `web` taint, and the policy refuses.

## Why a gateway, not an LLM-side guard

The model is adversarially-influenced by tool outputs. Anything we ask the model to do as
self-defense is bypassable. The gateway is outside the model's control surface, so the
guarantee is structural rather than emergent.


## Declassification

Some tools are trusted to *strip* a source label from their output. A vetted PII
redactor that scrubs identifiers should be allowed to remove the `pii` label;
without an explicit declassification mechanism, the only escape from a once-tainted
flow is to refuse it forever, which makes the gateway useless in practice.

We model this as a `ToolTaintSpec(adds, declassifies)`: every call's output label
is `((∨ inputs) ∨ adds) \ declassifies`. Declassification is a privilege a tool
declares once and the operator audits; it is intentionally *not* something the
LLM can request at runtime.

## Open questions

- Field-level taint inside structured tool outputs.
- Policy-as-code (Python) vs. data (YAML/JSON). Currently leaning data with a small set
  of well-defined operators.
- Streaming tool outputs and incremental taint propagation.

## Reference monitor (R4)

The runtime entry point is :class:`Gateway`, a small mutable container of
policies + tool taint specs + an optional audit-log writer. Two methods
matter:

- ``execute(call, fn, *args, resource=None, **kwargs) -> (value, Decision)``
  is the workhorse. It builds a :class:`Decision` from the policies, calls
  the audit writer (so even denials are recorded), and either invokes
  ``fn`` or raises :class:`PolicyDenied` / :class:`PolicyReview`. Both
  exceptions carry the original ``ToolCall`` and ``Decision``.
- ``wrap_tool`` is sugar over ``execute`` for the common case where you
  have a Python function and want every invocation gated. The wrapper
  recognises four reserved kwargs — ``apg_input_label``, ``apg_agent_id``,
  ``apg_call_id``, ``apg_resource`` — and strips them before forwarding
  the rest. ``resource_arg="url"`` ties policy resource matching to a
  named parameter of the wrapped function so the wrapper can pull a value
  out of either positional or keyword form via ``inspect.signature``.

Policy-action mapping at runtime:

- ``allow`` → :attr:`Verdict.ALLOW`, function runs.
- ``deny``  → :attr:`Verdict.DENY`, ``PolicyDenied`` raised, function does not run.
- ``review`` → :attr:`Verdict.REVIEW`, ``PolicyReview`` raised. We treat
  review as a hard refusal until a reviewer is wired in (a later
  milestone); the decision is preserved on the exception so callers can
  defer rather than abort if they want.
- ``rate_limit`` → :attr:`Verdict.ALLOW` for now. The DSL accepts a
  positive ``limit_per_minute`` and the rule_id is recorded in the
  decision, but no counter is enforced. R5 (audit log) introduces the
  per-tool window needed to make this a true throttle.

Cross-policy ordering: gateways may carry multiple policies. The first
policy with a matching rule wins; within a policy, the first matching
rule wins. This keeps composition predictable: appending an organisation
default after a stricter team policy never weakens the team policy.

Audit writing happens *before* the underlying function is invoked, so a
raising audit writer aborts the call. This is the "fail closed on audit"
posture: if you can't log a decision you don't get to act on it.


## MCP adapter (R6)

The MCP adapter (`agent_policy_gateway.mcp_adapter.wrap_mcp_session`) lets a
caller mount every tool advertised by an MCP-compatible session under a
`Gateway` in one line:

```python
tools = wrap_mcp_session(gateway, mcp_session)
tools["search"](query="apg", apg_input_label=TaintLabel.of("user"))
```

The session is duck-typed against a small *synchronous* protocol — `list_tools()`
returning either a bare iterable or a `.tools`-bearing wrapper, and
`call_tool(name, arguments)`. We deliberately do **not** import the real
`mcp` package: keeping the dependency surface narrow makes the adapter
testable against ad-hoc fakes, and keeps the project shippable to
environments that have not adopted the MCP SDK. The async transport is
the right shape for R9 (the dedicated async-gateway-path milestone), so
this milestone leaves it on the table.

`prefix` namespaces multiple sessions on the same gateway. Each tool is
registered as `<prefix>.<advertised>`, but the underlying MCP call always
uses the advertised (un-prefixed) name — the prefix is local to the
gateway's view, not something the server sees. `taint_specs` and
`resource_args` are keyed by the advertised name; the registered name is
what shows up in audit records and policy selectors, which are the two
places where prefixes need to be visible.


## OpenAI function-calling adapter (R7)

The OpenAI adapter (`agent_policy_gateway.openai_adapter`) splits the
function-calling integration into three small, composable pieces:

```python
from agent_policy_gateway import (
    Gateway, OpenAITool, openai_tool_specs, wrap_openai_tools,
    dispatch_openai_tool_call,
)

tools = [
    OpenAITool("search", "search the web", SEARCH_SCHEMA, search_fn,
               taint_spec=ToolTaintSpec.of(adds=("web",))),
    OpenAITool("send_email", "send mail", SEND_SCHEMA, send_fn,
               resource_arg="to"),
]

# 1) JSON tool descriptors for the API request payload.
api_tools = openai_tool_specs(tools)

# 2) Gateway-mediated callables, keyed by registered name.
wrapped = wrap_openai_tools(gateway, tools)

# 3) Convert each model-produced tool_call into a `role="tool"` message.
for tc in response.choices[0].message.tool_calls:
    chat.append(dispatch_openai_tool_call(
        wrapped, tc,
        input_label=TaintLabel.of("user"),
        agent_id="agent.research",
    ))
```

The OpenAI Python SDK is intentionally **not** a dependency — `tool_calls`
are duck-typed against either the dict shape returned by raw HTTP
responses (`{"id": ..., "function": {"name": ..., "arguments": "<json>"}}`)
or the attribute shape produced by the SDK's pydantic models. The adapter
is testable with hand-rolled fakes and shippable to environments that
have not adopted the `openai` package.

The reserved gateway kwargs (`apg_input_label`, `apg_agent_id`,
`apg_call_id`, `apg_resource`) are **not** advertised in the JSON
schemas — they are orchestration concerns, not function parameters. The
caller controls them via the `dispatch_openai_tool_call(...)` keyword
arguments; `apg_call_id` is auto-populated from the OpenAI `tool_call.id`
so audit records line up with the model's view of the conversation.

Refusal handling is structural: a `PolicyDenied` or `PolicyReview` raised
inside the wrapped callable is converted into a `role="tool"` message
whose content is a structured JSON payload
(`{"error": "policy_refusal", "verdict": ..., "rule_id": ..., "reason": ...}`).
That keeps the model in the loop — it sees a feedback signal it can
react to, rather than the orchestration loop terminating mid-turn.
Callers who want a different shape pass `on_denied=...` to
`dispatch_openai_tool_call`; tool exceptions and malformed model output
intentionally raise so they cannot be papered over.

Schema validation against `OpenAITool.parameters` is deferred. The model
typically produces conforming output, and any malformed argument surfaces
as a `TypeError` from the underlying Python function (still gateway-audited).
A future milestone can layer in `jsonschema` validation in front of
dispatch without changing this contract.

## Docs site (R11)

The project documentation is published via [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).
Configuration lives in `mkdocs.yml` at the repo root; site sources live under
`docs/` and double as input to both the rendered site and the in-repo
markdown reading flow (the milestone-design notes are exactly the same
files in either context). Contributors install the optional `docs` extra
(`pip install -e ".[docs]"`) and run `mkdocs serve` for a live-reloading
preview.

Two hard rules keep the site honest:

- The `nav` block in `mkdocs.yml` only references files that actually live
  under `docs/`. A dedicated test (`tests/test_docs_site.py`) walks the nav
  recursively and fails if a referenced file is missing — drift between
  reorg commits and `mkdocs.yml` therefore breaks CI rather than the
  rendered site.
- Build dependencies are pinned to the optional `docs` extra so the core
  test matrix stays dependency-light. The same test conditionally imports
  `mkdocs` and runs `mkdocs build --strict` into a tmp dir; if mkdocs is
  not installed (most CI matrix entries), it skips, otherwise it asserts
  the site builds without warnings.


## Benchmark harness (R12)

Performance work needs a baseline. `agent_policy_gateway.bench` ships a
small, dependency-free harness (`benchmark` + four canonical scenarios:
raw-call, gateway-allow, gateway-deny, gateway-allow-with-audit) that
reports min / mean / p50 / p95 / p99 / ops-per-sec for each scenario.
The same logic backs the `apg-bench` console script (`--iterations`,
`--warmup`, `--scenario`, `--json`).

A few decisions worth recording:

- **Per-call timing, not wall-clock-divided averages.** Each call is
  bracketed with `time.perf_counter_ns()`, so percentiles reflect real
  per-call latency. Mean is derived from the timed total.
- **Scenarios hold the tool body constant.** Every scenario invokes the
  same trivial `x + y` function, so the gateway-allow vs raw-call delta
  is mostly the gateway. Heavier tools would otherwise dominate the
  numbers and obscure overhead changes between releases.
- **No I/O in the default audit scenario.** `gateway_allow_audit` uses
  an in-process no-op `AuditWriter` so the comparison isolates
  audit-call dispatch cost from `JsonlAuditWriter`'s JSON-encode +
  fsync cost. The latter is intentionally not in the default suite —
  its variance across hosts and filesystems would make cross-host
  comparison noisy.
- **Tests assert structure, not thresholds.** `tests/test_bench.py`
  checks percentile ordering (`min <= p50 <= p95 <= p99 <= max`),
  scenario semantics (deny actually denies; audit writer actually
  fires), table/JSON shape, and CLI behaviour. Deliberately no
  absolute-microsecond assertions, because those flake on shared CI
  hosts and tell you nothing about whether the gateway code regressed.


## Threat model (R13)

The project's threat model is a project-level promise rather than a
per-feature appendix, so it lives in its own file
([`threat-model.md`](threat-model.md)) and is referenced from
feature docs rather than restated inline. Two consequences worth
recording:

- **Single source of truth.** When a new tool, adapter, or policy
  feature ships, the relevant adversary class and abuse scenario
  belong in `threat-model.md`, not in this design note. This page
  describes the *runtime mechanics* of the reference monitor; the
  threat-model page describes *who the runtime is up against and
  why*. A reader who wants both opens both.
- **Roadmap-worthy edits.** Substantive changes to assets,
  adversary classes, assumptions, or the in-scope/out-of-scope
  split are roadmap-worthy events: open a roadmap item, ship the
  change with code or tests, link the commit. Editorial fixes
  (typo, link, clarification) do not need a roadmap item. The
  test in `tests/test_threat_model.py` enforces only the
  structural invariants — section presence, nav inclusion,
  internal-link integrity — so editorial freedom is preserved.


## Release runbook (R14a)

R14 was split into two roadmap items:

- **R14a — Release runbook + manual fallback (docs).** Shipped
  with this commit. The runbook at
  [`release.md`](release.md) documents the steady-state release
  procedure (tag-and-push), the one-time PyPI trusted-publisher
  configuration, the `python -m build` + `python -m twine upload`
  manual fallback, and the post-release verification checklist.
  It is intentionally written so the manual fallback is a complete
  release path on its own — the project can cut a release today,
  before any GitHub Actions workflow lands.
- **R14b — Publish workflow (`.github/workflows/publish.yml`).**
  Still under *Up next* in the roadmap. The split exists because
  pushing a file under `.github/workflows/` requires a GitHub PAT
  carrying the `workflow` scope (classic) or *Workflows: Read and
  write* (fine-grained); the daily-task PAT currently does not
  carry that scope, so R14b cannot land via the scheduled run. The
  runbook ships first so the *procedure* is settled and reviewable
  independently of the *automation* that implements it.

Two design notes worth recording:

- **Trusted publishing is the destination, not a stored API token.**
  The runbook describes how the eventual workflow will use PyPI's
  OIDC flow (`permissions: id-token: write` + the `pypi` GitHub
  environment) so that no long-lived `PYPI_API_TOKEN` is stored
  in the repository or in GitHub secrets. The manual fallback
  uses a project-scoped API token only because `twine upload`
  has no OIDC mode; that token never leaves the developer's
  machine.
- **Test scope matches commit scope.** Today's commit ships
  `tests/test_release_runbook.py`, which asserts only what is
  already on disk: the runbook file, its mkdocs nav entry, the
  README and `docs/index.md` cross-links, the R14a design note,
  and the integrity of every relative link from the runbook.
  Workflow-file invariants (`v*` trigger, `test → build → publish`
  job graph, `id-token: write` permission, the `pypi`
  environment, the absence of a `PYPI_API_TOKEN` secret) land in
  `tests/test_publish_workflow.py` with R14b.


## Publish workflow (R14b)

R14a shipped the release *runbook*; R14b ships the GitHub Actions
*workflow* that runbook has been pointing at
(`.github/workflows/publish.yml`).
The workflow triggers on a `v*` tag push and on
`workflow_dispatch`; jobs `test → build → publish` run in sequence;
`publish` uploads via PyPI trusted publishing
(`pypa/gh-action-pypi-publish@release/v1`,
`permissions: id-token: write`, GitHub environment `pypi`). Three
design decisions are worth recording:

- **Artifact-upload-then-download between `build` and `publish`.**
  `build` uploads `dist/` as an `actions/upload-artifact@v4` artifact;
  `publish` downloads the same artifact via
  `actions/download-artifact@v4`. The alternative — running
  `python -m build` again inside `publish` — would produce a
  *different* sdist/wheel (different mtimes, different hashes) than
  the one `twine check --strict` inspected. The two-step hand-off is
  the only way to guarantee the bits twine checked are the bits that
  PyPI receives.
- **Top-level `permissions: contents: read`.** The workflow declares
  the minimum-privilege default at the top, and only the `publish`
  job opts into `id-token: write` (for the OIDC handshake). `test`
  and `build` cannot mint OIDC tokens, write to the repository, or
  touch GitHub's API beyond a read of the checked-out tree.
- **Structural tests, never version pins.** `tests/test_publish_workflow.py`
  asserts the workflow *shape* — triggers, job names, dependency
  graph, the `id-token: write` permission, the `pypi` environment
  name, the absence of any `PYPI_API_TOKEN` reference — and never
  pins an action version. Upgrading `actions/checkout` from `v4` to
  `v5` (or whatever) is editorial work that should not require
  rewriting the test suite, and the meaningful invariants
  (trusted-publishing path, three-stage gate) survive a version
  bump unchanged. The single end-to-end test that *does* execute
  real commands (`python -m build` + `twine check --strict`) is
  guarded by an import check on `build` and `twine` so the core
  test matrix stays dependency-light.

## Redact / declassify action (R17)

The [Declassification](#declassification) section above describes
declassification as a *tool* privilege: a vetted redactor declares
`ToolTaintSpec(declassifies={"pii"})` and the gateway strips the label
from that tool's output. R17 adds the *policy-driven* counterpart — a
`redact` effect that lets an operator declassify a flow at the policy
layer, transforming the call's arguments in place rather than trusting a
dedicated redactor tool to exist.

- **A third disposition between allow and deny.** Before R17 a matched
  rule could only let a call through untouched or refuse it. `redact`
  fills the gap the original Declassification note anticipated: the
  sensitive substring is masked and the now-clean value proceeds, so a
  once-tainted flow has an escape that is neither "refuse forever" nor
  "trust blindly". The operator declares it in the policy and audits it
  there; it is still *not* something the LLM can request at runtime.

- **Mask the downstream args and the audit args, in lock-step.** The
  transformation is applied twice from one `RedactSpec`: to the
  arguments the wrapped tool actually receives (bound through the
  function's signature so a positionally-passed field is reached the
  same as a keyword one, with a kwargs-only fallback when a callable
  cannot be introspected) and to the audit-visible `call.args`. Masking
  the audit copy is the point, not a side effect — an audit log that
  recorded the raw value would defeat the redaction. `redacted_fields`
  on the `Decision` records *that* redaction happened and *which* fields
  were touched, so a reviewer sees the rule id and the scope without the
  payload.

- **REDACT proceeds like ALLOW.** A new `Verdict.REDACT` makes the
  timeline honest (the replay tool shows `REDACT`, not `ALLOW`), but the
  execute paths treat it as a non-refusal: only `DENY` and `REVIEW`
  raise. `Decision.redacted_fields` is serialized only when non-empty so
  every pre-R17 audit record and the common allow/deny path keep their
  exact prior shape and round-trip unchanged.

- **Pattern semantics.** With a `pattern` set, only matching substrings
  inside a string field are masked (surrounding text is preserved) and
  non-string values are left untouched — a regex cannot meaningfully
  match them. With no pattern the whole field value is replaced by
  `mask`. The pattern is compiled at policy-load time so a bad regex is
  a load error, not a runtime surprise on the first matching call.

- **Hash-chained audit log is opt-in, not on by default (R27).** The
  tamper-evidence chain (`prev` = SHA-256 of the previous serialized
  line) is gated behind `JsonlAuditWriter(chain=True)` rather than
  enabled for every writer. The alternative — on-by-default with
  legacy tolerance — would change the default on-disk record shape and
  force every existing audit test and every already-written log into a
  migration. Opt-in keeps the legacy record byte-identical (the `prev`
  field is serialized only when set, mirroring the `redacted_fields` /
  provenance precedent) so `read_audit` still parses old logs and the
  R5 test suite stays green untouched; a deployer who wants
  tamper-evidence asks for it explicitly, exactly as they already opt
  into `fsync=True` for durability. The genesis sentinel is a fixed
  64-char all-zero string — same width as a real digest, but never a
  value `sha256` can produce, so "first record" is unambiguous. The
  running digest is seeded from the log's last line on open, so a chain
  survives writer restarts (append-only is the whole point) instead of
  silently forking at every reopen.

- **`--verify` is a chain walk, not a record parse (R27).** Tail
  truncation that lops whole records off the end leaves a *valid* prefix
  chain, so hash-chaining alone cannot prove a log is complete — that
  needs a signed checkpoint or a length oracle, out of scope here. What
  the chain *does* catch is any edit, reorder, deletion, or mid-line
  truncation, because each of those changes a line's bytes and breaks the
  next record's `prev`. `verify_chain` therefore reports the first line
  whose `prev` disagrees with the running digest (or whose JSON is
  unparseable, which a mid-line truncation produces), and `apg-replay
  --verify` surfaces that line number with a distinct exit code (4) so a
  caller can tell "chain broken" apart from "file missing" (2) without
  scraping stderr.

## Audit-log diff (R47)

- **Printed deltas are the difference of the printed shares, not of the
  exact ratios.** `apg audit diff` renders each verdict as
  `2 -> 1 (-1)  50.0% -> 20.0% (-30.0)`, and the parenthesized share
  delta is computed from the two one-decimal numbers on the same line
  rather than from the underlying ratios. A user checking the arithmetic
  on the page should always get the printed answer; a line that reads
  `33.3% -> 16.7% (-16.7)` would be internally inconsistent by one tick.
  The CI gates keep the opposite convention on purpose:
  `audit_flagged_share` / `audit_allow_share` threshold the *exact*
  share, because a value a hair over the limit must trip the gate even
  when it rounds down to the printed figure. Display rounds; decisions
  do not.

- **Rank movement, not just count movement.** The interesting question
  after a policy change is rarely "did `send_email` get called four more
  times" — traffic volume drifts on its own — but "did it become the
  thing we deny most". So each rules/tools/agents entry carries both
  sides' 1-based rank in the count-ordered list along with the raw
  counts, and `rank_delta` is `old_rank - new_rank` so a *positive*
  number reads as "climbed". Names present on only one side get a
  `None` rank and an `added` / `removed` status rather than a fake rank
  of 0, and they sort ahead of every mover: an entirely new rule firing
  is a bigger signal than an existing one shifting a place. Entries
  whose count *and* rank are unchanged are dropped, so two identical
  logs produce an empty movement list instead of a wall of zeros.

- **Diffing is not a gate.** `audit diff` has no `--fail-*` flag and
  exits `0` whether or not anything moved, mirroring `policy diff`.
  Thresholding a *delta* needs a policy about acceptable drift that the
  project does not have yet; until it does, `audit stats --fail-over` /
  `--fail-under` remain the CI surface, and `diff` stays a reporting
  tool.

## AgentDojo adapter (R49a)

- **Refusals return, they don't raise.** AgentDojo's pipeline default is
  `raise_on_error=False`, where a tool failure is returned to the model
  as an `"ErrorType: message"` string and the episode continues. The
  gated runtime folds policy refusals into exactly that convention
  (`"PolicyDenied: refused by rule '<id>': <reason>"`, rule id included
  so the R50 benchmark can attribute refusals from the transcript
  alone). A defense that crashed the episode on the first denied call
  would make utility-under-defense unmeasurable — the model must get
  the refusal as feedback and be free to try a legitimate path.

- **Taint accumulates per session, not per call.** Every other adapter
  lets the orchestrating code thread `apg_input_label` through each
  call; AgentDojo's LLM loop cannot. The wrapper therefore keeps a
  cumulative label: each executed call's `decision.output_label`
  (input ∨ adds, minus declassifies) becomes the next call's input
  label. This is deliberately conservative — once untrusted content
  enters the conversation, everything after it is treated as
  potentially influenced, which is precisely the indirect-prompt-
  injection threat model (A1 in `docs/threat-model.md`). Denied and
  errored calls contribute nothing (their output never reaches the
  model), a declassifying spec drops its sources, and `reset_taint()`
  starts a fresh episode. Finer per-message tracking is Phase 2
  territory (R51/R53).

- **Duck-typed, like every adapter.** `agentdojo` is not a dependency;
  the wrapper is written against the verified `FunctionsRuntime`
  surface (`functions` / `register_function` / `run_function`
  returning `(result, error | None)`) and everything else delegates
  through `__getattr__`, so the gated runtime is a drop-in wherever
  the real runtime was used. The real-package wiring lands with R49b
  behind an optional extra.

## AgentDojo suite wiring (R49b)

- **Reader classification is vector-derived and shipped as data.** A tool
  is an *untrusted reader* when its return value can carry
  attacker-authored text, determined from where each suite's injection
  vectors live in the benchmark's environment data (`data/suites/*`):
  banking places vectors in files and incoming-transaction subjects,
  slack in webpages and an externally-created channel name (plus message
  bodies authored by other workspace users), travel in
  hotel/restaurant/car-rental reviews, and workspace in calendar event
  bodies, cloud-drive files, and received emails. The tables are plain
  frozensets in `agentdojo_suite.py` — the policy knob R50 will measure —
  and alternative classifications go straight to
  `wrap_agentdojo_runtime` without touching the module. The integration
  tests assert the tables are subsets of the real suites' tools, so an
  upstream rename fails loudly.
- **One shared deny surface, deduplicated across suites.**
  `policies/agentdojo.yaml` carries one deny rule per sink in the
  cross-suite union (22 rules; `send_email` and the calendar mutations
  appear in both travel and workspace but get one rule each), every rule
  conditioned on `taint.any_of: [agentdojo:untrusted]`. Untainted calls
  match nothing and fall through to default-allow, so episodes that never
  read untrusted content keep full utility — the utility cost of the
  defense comes only from tasks that must read injectable content before
  a sink (e.g. banking's "pay this bill" reads the bill file first), which
  is precisely the tension R50 quantifies. The test suite keeps the YAML
  in lockstep with `_EXTERNAL_SINKS`.
- **`gate_suite` imports agentdojo only to build a runtime.** Tables and
  helpers stay importable without the benchmark; the lazy
  `FunctionsRuntime` import runs only when no `runtime=` is supplied, and
  a runtime missing classified tools raises (version-drift signal, tables
  target v1.2.1). Audit records default to `agentdojo:<suite>` so
  multi-suite logs split cleanly on the R36 `--agent` filter.

## AgentDojo benchmark: no defense vs APG (R50)

- **Scripted replay isolates the policy layer.** The benchmark replays
  ground-truth call sequences (user task's calls, then the hijacked
  agent's) with no LLM in the loop, so the numbers are deterministic,
  free, and attribute every delta to the gateway's per-call decisions:
  the no-defense arm is "what a fully hijacked agent achieves
  unimpeded", the apg arm is what survives `policies/agentdojo.yaml`.
  End-to-end LLM measurement (attacks that only sometimes land, models
  refusing on their own) is deliberately left to the R55 harness.
- **ASR counts armed episodes only.** 9 of the 35 default injection
  tasks (1 travel, 8 workspace) have empty scripted ground truths —
  they exfiltrate through the model's *output text*, which a tool-call
  gateway alone cannot see — and slack's `injection_task_3` attacks via
  `get_webpage`, a classified *reader*. The headline ASR therefore
  counts only episodes whose script contains an attack call on a
  classified sink ("armed"), excluding structurally unmeasurable pairs
  instead of letting them deflate both arms; the looser any-attack-call
  rate is reported alongside (it is what keeps slack's defended any-call
  figure at 60% — the reader-as-exfiltration channel R53's
  provenance-aware rules are meant to close).
- **Fresh environment per episode, deterministic episode ids.** Episodes
  mutate suite environment state (money moves, messages send), so each
  (user × injection) pair gets its own
  `load_and_inject_default_environment({})`; ids are
  `<suite>:<user_task>x<injection_task>:<arm>` so the two arms' JSONL
  files line up row-for-row for offline diffing.
- **Headline result (agentdojo 0.1.35, suites v1.2.1).** Sink-level ASR
  drops 100% → 0% on all four suites; utility costs are banking
  100→25%, slack 100→4.8%, travel 100→70%, workspace 100→55%. The
  spread *is* the Phase 2 motivation: one blanket "no tainted sink
  calls" rule buys complete (measurable) attack suppression at a
  utility price that R51 dual labels, R52 declarative declassify, and
  R53 chain-level rules exist to bring down. Banking integration tests
  pin these numbers against `docs/benchmarks/agentdojo.md`, so a policy
  or classification change that moves them fails the suite.

## Dual-label taint: confidentiality + integrity (R51)

- **Two dimensions, one atom set.** A `TaintLabel` now carries three
  frozensets: `confidentiality` (secret data that must not reach public
  sinks), `integrity` (untrusted data that must not drive privileged
  actions), and the legacy `sources` set, which counts in **both**
  dimensions. That last rule is the whole back-compat story: every
  pre-R51 label, spec, policy, and audit record behaves identically,
  because a single-set label's effective confidentiality and integrity
  sets are both just `sources`. The R50 benchmark's blanket policy is
  untouched; R51 only makes finer policies *expressible*.
- **Canonical form makes equality semantic.** A source present in both
  dimension sets means exactly what membership in `sources` means, so
  `__post_init__` promotes it (and drops dimension entries shadowed by
  `sources`). `TaintLabel.of("x")` and
  `of_dimensions(confidentiality=["x"], integrity=["x"])` are therefore
  `==`, joins never produce a non-canonical label, and serialization is
  stable — the dimension keys are emitted only when non-empty, so legacy
  records keep their JSON shape byte-for-byte.
- **Propagation is per-dimension; declassification splits into
  declassify and endorse.** `join` / `subsumes` / `propagate` operate on
  the per-dimension effective sets. `ToolTaintSpec.adds` /
  `.declassifies` are themselves labels, so a spec can add or strip a
  source in one dimension only: stripping from confidentiality is
  declassification proper (a vetted PII redactor — the secret is gone,
  the untrustedness stays), stripping from integrity is *endorsement* in
  the IFC-literature sense (a sanitizer vouching the content can no
  longer steer the agent — the secrecy stays). A redact effect's
  `declassify` remains a full strip across every dimension: the masked
  field can neither leak nor steer.
- **Policy clauses: top-level = union, nested = one dimension.** The
  existing `taint.any_of/all_of/none_of` clauses now match the union of
  dimensions (identical for legacy labels, and the fail-closed reading
  for dimension-scoped ones), while nested `taint.confidentiality:` /
  `taint.integrity:` sub-conditions match a single dimension's effective
  set. This is what separates the two AgentDojo failure modes: "deny
  `send_money` when integrity-tainted" no longer fires on merely-secret
  data, and "deny `post_public` when confidentiality-tainted" no longer
  fires on merely-untrusted data. `policy lint` W002 understands the new
  clauses (including cross-checks against the top-level `none_of`);
  the W001 shadow check stays deliberately conservative — a rule with
  dimension sub-conditions never claims generality, trading missed
  warnings for zero false ones.

## Declarative declassify (R52)

Until R52, declassification was a *code-level* privilege: any code path
that could register a `ToolTaintSpec` could also declare
`declassifies=...` and silently launder taint — the policy file, the
artifact that is reviewed, versioned, and diffed, had no say in it. R52
moves the authority into the policy: a top-level `declassify:` section
lists **grants**, each naming which tool (fnmatch glob) may strip which
sources (globs over source names) from which label dimensions, under
what conditions:

```yaml
declassify:
  - id: sanitizer-endorses-web
    tool: sanitize_html
    sources: [web]
    dimensions: [integrity]        # endorsement; default is both dimensions
    identity: agent.research       # optional
    resource: "https://trusted/*"  # optional
    when:                          # optional condition on the input label
      confidentiality:
        none_of: [pii]
```

Semantics, chosen deliberately:

- **Presence of grants flips the authority.** A gateway is
  *declassification-governed* iff any loaded policy carries a non-empty
  `declassify:` section. Ungoverned gateways behave byte-for-byte as
  before (per-spec `declassifies` applies) — the entire back-compat
  story, mirroring R51's. Governed gateways treat the policy as the
  *sole* authority: `ToolTaintSpec.declassifies` is inert, and the
  output label is the raised label (inputs ∨ `spec.adds`) minus what
  matching grants permit. Migrating a spec-declared strip is a
  three-line grant.
- **Grants strip directly; all matching grants contribute.** Unlike
  rule matching there is no first-match: strips union across every
  matching grant of every policy, in declaration order. A grant strips
  only from the dimensions it lists — `dimensions: [integrity]` is
  endorsement (untrustedness gone, secrecy kept), `[confidentiality]`
  is declassification proper — reusing the R51 vocabulary.
- **Fired grants are audited.** `Decision.declassified_by` records the
  ids of grants that actually removed at least one source (a matching
  grant that stripped nothing is not recorded), serialized only when
  non-empty so legacy record shapes are unchanged; `apg-replay` prints
  them as a `declassify:` line. Provenance entries for stripped sources
  drop with the sources, as with every declassification.
- **Redact effects are unchanged.** A `redact` effect's `declassify`
  was already policy-declared, so it still applies under governance —
  after any grant strips.
- **Tooling.** `apg policy explain` appends a grant-by-grant match
  trace whenever the policy declares grants (grantless policies render
  byte-for-byte as before); `apg policy lint` extends W002 to grants
  whose `when:` condition is self-contradictory and adds **W003** for
  an unconditional strip-everything grant (`tool: "*"`,
  `sources: ["*"]`, both dimensions, no identity/resource/when) — legal
  but almost certainly a policy bug. `WatchedPolicy` (R20) grew the
  `declassify` / `matching_grants` duck-type members, so hot-reloaded
  policies govern like plain ones.

`policies/declassify-sanitizer.yaml` is the runnable example: an
integrity-only endorsement grant for `sanitize_html`, and a
`pii_redactor` grant conditioned on the *absence* of web taint — a
redactor fed attacker-authored text must not launder it.

## Chain-level policies (R53)

The label model (R2–R51) answers "what does this value carry?"; some
policies need to answer "what has this session *done*?". The motivating
gap is the R50 slack residue: `get_webpage` is an untrusted reader, not
a sink, so a hijacked agent can exfiltrate by *fetching* an
attacker-controlled URL that encodes stolen data — no rule over the
current call's input label distinguishes that fetch from the first,
legitimate one. R53 adds a `chain:` sub-condition to `Selector`, stated
over the session's call history and the input's provenance chain:

```yaml
- id: deny-send-after-web
  when:
    tool: send_email
    chain:
      any_prior:                 # >=1 recorded prior call matches >=1 matcher
        - source: web            # glob over that call's *output label*
          verdict: allow         # only executed calls arm this rule
      no_prior:                  # no recorded prior call matches any matcher
        - tool: sanitize_*
      provenance:                # condition on the input's provenance chain
        any_of:
          - {source: web, tool: "get_*"}
  effect: {action: deny}
```

Semantics, chosen deliberately:

- **History is gateway state, opt-in.** `Gateway(track_history=True)`
  records one `CallHistoryEntry` per mediated call — tool, verdict,
  output label, call id, resource — keyed by `agent_id`, appended on
  the execute paths only, *after* the decision is final, so a call
  never sees itself in its own history. The pure `decide()` reads
  history but records nothing (the rate-limiter peek/consume
  precedent). `call_history()` / `reset_history()` expose and clear it;
  the durable record remains the audit log.
- **Denied attempts are recorded.** A chain policy may care that an
  agent *tried* something ("no retries after a refusal"), so every
  verdict lands in history; the matcher's `verdict:` field scopes a
  rule to executed calls (`verdict: allow`) when only real effects
  matter. Matching on a prior call's `source:` globs its **output
  label** across every dimension — "a call that returned web content
  preceded this one".
- **Untracked history fails closed at the selector.** `history=None`
  (a gateway without `track_history`) is distinct from an empty
  history: a chain condition referencing prior calls does not match at
  all — `no_prior` included, since "no forbidden call happened" cannot
  be verified without a record — exactly the `Selector.resource`
  precedent for an unsuppliable constraint. Chain-history decisions
  are therefore byte-for-byte pre-R53 unless a deployment opts in.
- **Provenance clauses ride the existing chains.** `chain.provenance`
  matches the current call's `input_provenance` entries (R30) —
  source/tool matcher globs under `any_of`/`none_of` — needing
  `track_provenance`, not `track_history`. Where the history clauses
  say "somewhere in this session", provenance says "in *this value's*
  derivation" — the finer, flow-scoped statement.
- **Tooling.** `apg policy explain --prior 'get_webpage,source=web'`
  builds synthetic history (omitting `--prior` means an *empty*
  history, so traces mirror a tracking gateway); lint W002 extends to
  unsatisfiable chains (`any_prior` entirely forbidden by `no_prior`,
  same for provenance `any_of`/`none_of`) and W001 stays conservative
  (a chain-constrained rule never claims generality); `policy diff`
  scenarios carry no history, so chain rules are documented as outside
  the matrix's reach. `WatchedPolicy.first_match` mirrors the new
  `history` keyword.

The AgentDojo wiring closes the loop: the benchmark gateway now tracks
history (recording never changes a history-free policy's decisions),
the adapter's `reset_taint()` also resets it between episodes, and
`policies/agentdojo-chain.yaml` — the baseline plus
`deny-web-fetch-after-untrusted-read` — cuts slack's any-call ASR from
60% to 40% (the whole reader-borne channel) at zero utility cost; see
`docs/benchmarks/agentdojo.md`. On that scripted replay an input-taint
rule would score the same; the chain form survives declassification
and sees denied attempts, which a label cannot express.

## Progent rule import (R54)

Progent (Shi et al., "Progent: Programmable Privilege Control for LLM
Agents"; [`sunblaze-ucb/progent`](https://github.com/sunblaze-ucb/progent))
guards agent tool calls with per-tool symbolic rules. Its runtime
representation — verified against upstream `secagent/tool.py` rather
than paraphrased from the paper — is
`{tool: [(priority, effect, condition, fallback), ...]}`: rules sorted
by `(priority, -effect)` (lower priority number first; at a tie,
*forbid* before *allow*), conditions mapping argument names to JSON
Schema fragments, bare `re.match` regex strings, or callables, and
fallbacks saying what a firing forbid does (`0` error message back to
the agent, `1` terminate the process, `2` ask the user). A tool with no
entry is denied, as is a call that satisfies no rule of its tool. R54
is a proof of concept that this rule language embeds into APG's ordered
first-match policy DSL: `apg policy import-progent` (and
`convert_progent_policy` under it) translates the JSON serialization of
that mapping into a plain APG policy — after which the entire existing
toolchain (explain traces, lint, decision diff, audit, the gateway
itself) applies to policies authored for a different system.

Decisions worth recording:

- **`Selector.arg_matches` is the load-bearing addition.** Progent's
  flagship rules are pattern-shaped ("`send_money` only to this IBAN"),
  which `arg_equals` cannot express. `arg_matches` maps argument names
  to regexes with `re.search` semantics — deliberately the JSON Schema
  `pattern` convention, so imported rules keep their meaning; Progent's
  bare-string shorthand (which upstream checks with `re.match`) is
  anchored as `\A(?:...)` at translation time. A constrained argument
  must be present and be a string; the empty pattern reads "any
  string", which is exactly how `{"type": "string"}` imports. Explain
  gained an `arg_matches` rejection trace, lint W002 catches an
  `arg_equals` literal that cannot satisfy the `arg_matches` regex on
  the same argument, and W001 stays conservative (identical patterns
  subsume, nothing else claimed — no regex-implication reasoning).
- **Order and defaults translate structurally.** Rules are emitted in
  Progent's sorted order; `enum`/`const` restrictions expand into one
  rule per value combination (capped at 64 — the import fails loudly
  above it); each governed tool gets a trailing fall-off-the-end rule
  and, in the default standalone mode, a global catch-all deny mirrors
  "a tool with no entry is denied" (`--default per-tool` omits the
  catch-all for merging into larger rule sets). Two upstream quirks are
  carried faithfully: the *hard allow* (an allow at priority 100 with
  fallback 0 denies immediately when a present argument fails, so its
  translation appends a tool-wide deny right after the allow), and the
  *leaky fallback* (falling off the end is handled with the last
  examined rule's fallback — so a tool whose last rule says "ask the
  user" gets a trailing `review`, not `deny`).
- **Effects map onto the existing action vocabulary.** Forbid+fallback
  `0` → `deny`; fallback `2` → `review` (APG's escalate-to-human verdict
  is precisely Progent's interactive confirmation, minus the terminal
  prompt); fallback `1` → `deny` with a reason noting that Progent
  would terminate — APG never calls `sys.exit` from a decision.
- **Unsupported means loud, never weaker.** Callables, five-element
  self-updating rules (`need_update_policies`), JSON Schema keywords
  beyond `const`/`enum`/`pattern`/`type: string`, and float/null
  scalars raise `ProgentImportError`; nothing silently degrades to a
  more permissive policy. Within the subset, fidelity is tested against
  a reference evaluator ported from upstream `_check_tool_call`
  (`tests/test_progent_import.py`), which must agree with the converted
  policy's first-match decision on a battery of policies × calls.
- **Documented divergences.** (1) Progent checks a restriction only
  when the argument is present — an allow rule's constraint on an
  absent argument passes vacuously. APG requires presence, so
  translated allow rules are *stricter* (fail closed) and translated
  forbid rules do not fire on absent arguments (the call then falls to
  the trailing default). (2) JSON Schema treats `pattern` as
  inapplicable to non-strings (vacuously valid); `arg_matches` never
  matches a non-string — again fail-closed for allows. (3) Terminate
  becomes deny. Each divergence has an explicit test.
- **`policy_to_yaml` is the other half of the PoC.** The converter's
  output is a *file*, not just an in-memory object: a generic
  `Policy` → YAML serializer (defaults pruned, round trip
  `load_policy_str(policy_to_yaml(p)) == p` pinned for the shipped
  example policies too) so converted policies are reviewable, lintable
  artifacts. `examples/progent/` runs the whole path — Progent JSON →
  YAML → gateway decisions — as a CI sanity check, and the R56
  comparison now has a mechanical way to run Progent-authored policies
  under APG.
