# Security policy

## Reporting a vulnerability

agent-policy-gateway is a security control, so bypasses matter. Report them
privately through GitHub Security Advisories: open the repository's
**Security** tab and select **Report a vulnerability**
(<https://github.com/howardhsieh/agent-policy-gateway/security/advisories/new>).
Do not open a public issue for a vulnerability.

Include the version or commit, the policy file, the sequence of tool calls
(synthetic data only), the decision the gateway made, and the decision you
expected. You can expect an acknowledgement within 7 days. Fixes ship as a new
release and are credited in the release notes unless you prefer otherwise.

## Scope

In scope:

- A tool call that a policy should deny or send to review but the gateway
  allows (policy-evaluation, taint-propagation, declassification or
  chain-rule bypass).
- Audit-log integrity: records that can be dropped, altered or forged without
  breaking the hash chain, or replay and export tools that misreport them.
- Adapters (MCP, OpenAI, Anthropic, LangChain, AgentDojo) that let a call
  reach a tool without mediation.
- Crashes or unbounded resource use from crafted policies or audit logs.

Out of scope: the residual risks documented in
[`docs/threat-model.md`](docs/threat-model.md) (for example exfiltration
through the model's own output text, where no tool call happens), and
vulnerabilities in the tools themselves.

## Supported versions

Only the latest release on PyPI receives fixes.
