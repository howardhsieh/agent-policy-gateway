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
