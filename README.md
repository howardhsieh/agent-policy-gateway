# agent-policy-gateway

A policy enforcement and information-flow-control (IFC) gateway for AI agent tool calls.

> **Status:** Pre-alpha. Active development. See [`ROADMAP.md`](./ROADMAP.md).

## Why

Autonomous AI agents increasingly call tools — read files, query databases, browse the web,
send messages. When data from an untrusted source (e.g. a web page) flows into a sensitive
sink (e.g. an email send), bad things happen: prompt injection becomes prompt *exfiltration*,
indirect commands turn into real-world actions, and audit trails are missing.

Most existing guardrails focus on *LLM output filtering*. `agent-policy-gateway` instead sits
between the agent and its tools and treats the problem the way operating systems treat untrusted
input: with **policy enforcement** and **taint tracking** across calls.

## What it does

1. **Policy enforcement.** Allow / deny / quota / human-in-the-loop rules per tool, per
   resource, per agent identity. Policies are declarative (YAML) and composable.
2. **Taint tracking.** Each tool call produces tainted output (sources: web, user-uploaded
   docs, external API). Taint propagates through subsequent calls. Policies can refuse a
   sensitive sink (e.g. `send_email`) if its arguments carry high-taint sources.
3. **Audit log.** Every call, decision, and taint label is recorded in an append-only log
   suitable for incident response and red-team review.
4. **Multi-protocol adapters.** Wraps tool catalogs from MCP, OpenAI function calling,
   Anthropic tool use — same policy language across them.

## Threat model

We assume the LLM itself is **not** trusted to decide what tool calls are safe. Adversarial
content can reach the model via any tool output, and the model may then attempt unsafe calls.
The gateway is the trusted reference monitor; the model is policy-controlled, not policy-aware.

The full project threat model — assets, trust boundaries, adversary classes, assumptions,
canonical abuse scenarios mapped to the mitigations the gateway already ships, and residual
risks — lives at [`docs/threat-model.md`](./docs/threat-model.md).

## Non-goals

- Not a content filter. We do not classify text as "harmful." This is about *flow*.
- Not a sandbox for tool *implementations*. Tools run as they always did; we mediate access.
- Not a replacement for human review of high-stakes actions. We make review tractable.

## Quick start

```python
from agent_policy_gateway import Gateway, Policy

gw = Gateway.from_yaml("policies/my-policy.yaml")

@gw.wrap_tool(name="web_search", taint_sources={"web"})
def web_search(query: str) -> str:
    ...

@gw.wrap_tool(name="send_email", taint_sinks={"web": "deny"})
def send_email(to: str, body: str) -> None:
    ...

# Now the agent calls these wrappers; the gateway enforces policy and tracks taint.
```

## Benchmarks

The `apg-bench` console script runs a small standard-library-only suite that
measures per-call overhead and throughput on the gateway's hot paths
(raw call, gateway-allow, gateway-deny, gateway-allow + audit). After
`pip install -e .` the script is on `$PATH`:

```bash
apg-bench                       # human-readable table
apg-bench --json                # machine-readable
apg-bench --scenario raw_call   # one scenario only
```

See [`docs/benchmarks.md`](./docs/benchmarks.md) for the methodology and
the public `benchmark()` / `run_default_suite()` API.

## Documentation

The full documentation site is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).
Build it locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

The site sources live under [`docs/`](./docs/) and the configuration in
[`mkdocs.yml`](./mkdocs.yml). Start with [`docs/index.md`](./docs/index.md) and
[`docs/quickstart.md`](./docs/quickstart.md).

## Release

Releases are published to PyPI. The intended automated path is a
GitHub Actions workflow at `.github/workflows/publish.yml`
(roadmap item **R14b**, still pending) that builds the sdist + wheel
and uploads them via PyPI's trusted-publisher OIDC flow — no
long-lived API token is stored in the repo. Until R14b lands, the
documented manual fallback (`python -m build` + `python -m twine upload`)
is the working path. The full procedure — one-time PyPI trusted-publisher
setup, the tag-and-push flow, the manual fallback, and a post-release
verification checklist — is in [`docs/release.md`](./docs/release.md).

## License

Apache-2.0. See [`LICENSE`](./LICENSE).

## Contributing

This project is built incrementally and in public. See [`ROADMAP.md`](./ROADMAP.md) for what's
next, and [`docs/design.md`](./docs/design.md) for the architecture.
