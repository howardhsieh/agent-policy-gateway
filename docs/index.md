# agent-policy-gateway

Welcome to the documentation site for **agent-policy-gateway** — a policy enforcement
and information-flow-control (IFC) gateway for AI agent tool calls.

> **Status:** Pre-alpha. Active development. See the [`ROADMAP`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/ROADMAP.md).

## What it is

`agent-policy-gateway` sits between an LLM-driven agent and the tools the agent calls.
It is a *reference monitor* for tool invocations:

- Every tool call passes through it.
- Every decision (allow / deny / review) is logged.
- Policies cannot be bypassed by content the model has read — including
  prompt-injected web pages or documents.

The gateway pairs **policy enforcement** with **taint tracking**: tool outputs carry
*source labels* like `web` or `user_upload`, taint propagates through downstream calls,
and policies can refuse a sensitive sink (`send_email`, `db.write`) when its arguments
carry untrusted taint.

## Where to start

| If you want to… | Read this |
| --- | --- |
| See it work in 30 lines of Python | [Quickstart](quickstart.md) |
| Understand the threat model and architecture | [Design](design.md) |
| Read the project threat model in full | [Threat model](threat-model.md) |
| Read the open-source roadmap | [ROADMAP](https://github.com/howardhsieh/agent-policy-gateway/blob/main/ROADMAP.md) |
| Follow the daily, autonomous build cadence | [Daily runbook](daily-runbook.md) |

## Adapters

`agent-policy-gateway` ships protocol adapters so you can mount the same policy across
your existing tool catalogue:

- **MCP** — `wrap_mcp_session` / `wrap_mcp_session_async` mount every tool a session
  advertises behind the gateway in one line.
- **OpenAI function calling** — `openai_tool_specs`, `wrap_openai_tools`, and
  `dispatch_openai_tool_call` plug into the Chat Completions / Responses request and
  reply pipeline.
- **Anthropic tool use** — `anthropic_tool_specs`, `wrap_anthropic_tools`, and
  `dispatch_anthropic_tool_use` produce the Messages-API `tools=[...]` array and turn
  model `tool_use` blocks into `tool_result` blocks (including a structured
  `policy_refusal` payload when the gateway blocks a call).

The real `mcp`, `openai`, and `anthropic` SDKs are intentionally **not** runtime
dependencies — every adapter is duck-typed against the on-the-wire shapes those
libraries already produce.

## License

Apache-2.0. See [`LICENSE`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/LICENSE).
