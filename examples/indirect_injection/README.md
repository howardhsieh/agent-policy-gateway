# Indirect prompt injection — worked example (R10)

This example demonstrates how `agent-policy-gateway` defends a tool-using
agent against an *indirect prompt injection* attack.

## The scenario

A research agent is asked to summarise a news page. The page it fetches has
been crafted by an attacker: alongside its visible content it carries an
injected instruction telling the agent to read an internal record and email
its contents to an attacker-controlled address.

We compare two configurations of the same agent against the same page:

| Variant | Tools                                  | Outcome |
|---------|----------------------------------------|---------|
| Naive   | Raw functions                          | Email is sent. Secret leaks. |
| Gated   | Wrapped by `Gateway` + `default.yaml`  | `send_email` is refused by `deny-web-to-email`. Secret stays put. |

The "agent" here is a deterministic stub that follows any "send its contents
to `<email>`" instruction it sees in fetched content. That captures the
failure mode of an LLM-driven agent without depending on a live model. The
defence works the same way against a real model — the gateway makes its
decision from the **taint label** on the tool call's inputs, not from
anything the agent says or thinks.

## Run it

From the repository root:

```
python -m examples.indirect_injection
```

Expected output:

```
naive: exfiltrated record 'launch-codes' (body='TOP-SECRET-LAUNCH-CODES-42') to attacker@evil.example
gated: refused by rule 'deny-web-to-email' — outbox empty, N audit record(s) written
```

Exit code is `0` when both variants behave as expected, `1` otherwise — so
you can run the example as a smoke test in CI.

## How the defence works

* `web_fetch` is registered with a `ToolTaintSpec` that adds the `web`
  source on every call. Any value derived from a `web_fetch` result inherits
  that label.
* `send_email` is wrapped through the same `Gateway`, so each call goes
  through `Gateway.execute`.
* `policies/default.yaml` contains:

  ```yaml
  - id: deny-web-to-email
    when:
      tool: send_email
      taint:
        any_of: [web]
    effect:
      action: deny
  ```

  The gateway sees that `send_email` is being called with an input label
  carrying `web` and refuses with `PolicyDenied`. The audit writer records
  the decision before the underlying function runs, so the refusal is
  visible to incident response.

## Files

* `injected_page.py` — the poisoned web page (HTML string + URL constant).
* `tools.py` — `web_fetch`, `kb_lookup`, `send_email` plus an in-memory
  outbox for observation.
* `agent.py` — the stub agent driver shared by both variants.
* `naive.py` / `gated.py` — the two runners.
* `__main__.py` — the single-command entry point.

The example imports `agent_policy_gateway` the same way an external project
would, so it doubles as a working integration sample.
