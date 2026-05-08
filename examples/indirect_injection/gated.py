"""Gated runner: the same agent, mediated by ``agent-policy-gateway``.

Configuration:

* Policies come from ``policies/default.yaml``. That policy ships in the
  repository and contains the ``deny-web-to-email`` rule, which is exactly
  the rule the gateway will fire when the injected page's content (carrying
  a ``web`` taint) is passed into ``send_email``.
* Tools are wrapped with :meth:`Gateway.wrap_tool`. ``web_fetch`` is given
  a ``ToolTaintSpec`` declaring that it intrinsically adds the ``web``
  source on every call, so any value derived from it carries that label
  forward.
* The agent driver passes ``apg_input_label`` through ``send_email`` so the
  gateway can compute the right input label for the call. This is how an
  agent runtime would normally thread taint state from one tool call to
  the next; in a real harness the runtime would track these labels
  automatically across the agent loop.

The runner returns enough information for both the example output and the
tests: the agent result, the outbox (which should remain empty), the audit
log records the gateway produced, and the gateway itself for tests that
want to inspect its policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent_policy_gateway import (
    Decision,
    Gateway,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    load_policy,
)
from examples.indirect_injection.agent import AgentResult, run_research_task
from examples.indirect_injection.tools import Outbox, build_tools

#: Path to the shipped default policy. Resolved at runtime from the repo
#: root so the demo runs whether the user is in ``examples/`` or at the
#: project root.
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "policies" / "default.yaml"
)


@dataclass
class _AuditLog:
    """Tiny in-memory audit-writer shim, suitable for the demo and tests."""

    records: list[tuple[ToolCall, Decision]] = field(default_factory=list)

    def __call__(self, call: ToolCall, decision: Decision) -> None:
        self.records.append((call, decision))


@dataclass
class GatedRunResult:
    """Bundle of the agent result, the outbox, and gateway/audit state."""

    agent: AgentResult
    outbox: Outbox
    gateway: Gateway
    audit: list[tuple[ToolCall, Decision]]


def build_gateway(policy_path: Path | str = DEFAULT_POLICY_PATH) -> Gateway:
    """Build a :class:`Gateway` configured with the shipped default policy.

    Exposed as a separate function so tests can construct a gateway without
    running the full demo.
    """
    policy = load_policy(str(policy_path))
    audit = _AuditLog()
    return Gateway(policies=[policy], audit_writer=audit)


def run_gated(policy_path: Path | str = DEFAULT_POLICY_PATH) -> GatedRunResult:
    """Run the demo with every tool call mediated by the gateway."""
    tools = build_tools()
    gateway = build_gateway(policy_path)
    audit: _AuditLog = gateway.audit_writer  # type: ignore[assignment]

    @gateway.wrap_tool(
        tool_name="web_fetch",
        taint_spec=ToolTaintSpec.of(adds=("web",)),
    )
    def web_fetch(url: str) -> str:
        return tools.web_fetch(url)

    @gateway.wrap_tool(tool_name="kb_lookup")
    def kb_lookup(record: str) -> str:
        return tools.kb_lookup(record)

    @gateway.wrap_tool(
        tool_name="send_email",
        taint_spec=ToolTaintSpec.of(),
        resource_arg="to",
    )
    def send_email(to: str, body: str) -> dict:
        return tools.send_email(to, body)

    # The agent driver does not (yet) thread taint labels for us, so we
    # supply a thin shim that calls the wrapped tools with the right
    # ``apg_input_label``. ``send_email`` carries the ``web`` label because
    # its body was derived from a ``web_fetch`` result; in a real agent
    # harness this bookkeeping would happen automatically.
    web_label = TaintLabel.of("web")

    def _web_fetch(url: str) -> str:
        return web_fetch(url, apg_agent_id="agent.research")

    def _kb_lookup(record: str) -> str:
        return kb_lookup(record, apg_agent_id="agent.research")

    def _send_email(to: str, body: str) -> dict:
        return send_email(
            to,
            body,
            apg_input_label=web_label,
            apg_agent_id="agent.research",
        )

    agent = run_research_task(
        web_fetch=_web_fetch,
        kb_lookup=_kb_lookup,
        send_email=_send_email,
    )
    return GatedRunResult(
        agent=agent,
        outbox=tools.outbox,
        gateway=gateway,
        audit=list(audit.records),
    )


__all__ = ["GatedRunResult", "build_gateway", "run_gated"]
