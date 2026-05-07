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

