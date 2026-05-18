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
([`.github/workflows/publish.yml`](../.github/workflows/publish.yml)).
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
