"""Naive runner: no gateway, no taint tracking, no policy.

This is the *vulnerable* configuration. The agent is handed the raw tool
functions and runs to completion — including the exfiltrating
``send_email`` call. The runner returns the agent result and the outbox so
callers (the ``__main__`` entry point and the tests) can confirm the
exfiltration happened.

We keep this runner here in the example, rather than as throwaway code,
because being able to *demonstrate* the failure mode against the same
drivers and the same poisoned page that the gated runner uses is half of
the value of the example.
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.indirect_injection.agent import AgentResult, run_research_task
from examples.indirect_injection.tools import Outbox, build_tools


@dataclass
class NaiveRunResult:
    """Bundle of the agent result and the outbox after a naive run."""

    agent: AgentResult
    outbox: Outbox


def run_naive() -> NaiveRunResult:
    """Run the demo with no gateway in front of the tools."""
    tools = build_tools()
    agent = run_research_task(
        web_fetch=tools.web_fetch,
        kb_lookup=tools.kb_lookup,
        send_email=tools.send_email,
    )
    return NaiveRunResult(agent=agent, outbox=tools.outbox)


__all__ = ["NaiveRunResult", "run_naive"]
