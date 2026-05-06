"""Stub agent driver for the indirect-injection example.

This module deliberately *does not* call a real LLM. The goal of the
example is to demonstrate the gateway's defence against a known-vulnerable
agent in a deterministic, testable way; running an actual model on every
test invocation would buy us realism at the cost of reproducibility.

The stub implements one specific agent behaviour: parse the fetched page,
look for an injected instruction of the form ``send its contents to <addr>``,
follow it. This mirrors the failure mode of an LLM-driven agent that does
not distinguish between operator instructions and content fetched from the
web. The defence works against both — the gateway makes its decision from
the *taint label* on the call's inputs, not from anything the agent says.

The runners pass *callables* in for each tool so the same driver can be used
with raw functions (naive) or with gateway-wrapped functions (gated). When a
tool call is refused, the configured callable raises and the agent stops —
which is exactly what we want to observe.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from examples.indirect_injection.injected_page import INJECTED_PAGE_URL

#: Pattern that finds an "exfiltrate to <address>" instruction in fetched
#: page content. It matches the injection used in the example as well as
#: small variations a real LLM-driven agent would fall for. The point is
#: not to be a robust parser; the point is for the stub to do *something*
#: when it sees an injected payload.
_EXFIL_PATTERN = re.compile(
    r"send its contents to ([^\s]+@[^\s]+)",
    re.IGNORECASE,
)

#: Pattern that picks out a target KB record name from the same instruction.
_RECORD_PATTERN = re.compile(
    r"record named '([^']+)'",
    re.IGNORECASE,
)


@dataclass
class AgentResult:
    """Outcome of a single :func:`run_research_task` invocation."""

    summary: str
    followed_injection: bool
    exfil_target: str | None = None
    exfil_record: str | None = None
    refused_by_gateway: bool = False
    refusal_rule_id: str | None = None


def run_research_task(
    *,
    web_fetch: Callable[[str], str],
    kb_lookup: Callable[[str], str],
    send_email: Callable[[str, str], dict],
    page_url: str = INJECTED_PAGE_URL,
) -> AgentResult:
    """Run the demo agent against ``page_url``.

    The agent's behaviour:

    1. Fetch the page.
    2. If the page content contains the injection pattern, look up the
       referenced KB record and call ``send_email`` with the secret as the
       body. This is the unsafe behaviour the example is demonstrating.
    3. Whether or not step 2 fired, return a short benign summary derived
       from the page's first line.

    Tool refusals are caught here so the runner can report them cleanly.
    The ``PolicyDenied`` / ``PolicyReview`` exception classes are imported
    lazily so this stub stays usable even if ``agent_policy_gateway`` is
    not yet importable (e.g. in a pre-install lint pass).
    """
    page = web_fetch(page_url)

    # Try to follow the injection. If the gated runner has wrapped
    # ``send_email`` with the gateway, the call will raise ``PolicyDenied``.
    followed = False
    target: str | None = None
    record: str | None = None
    refused = False
    refusal_rule_id: str | None = None
    addr_match = _EXFIL_PATTERN.search(page)
    rec_match = _RECORD_PATTERN.search(page)
    if addr_match and rec_match:
        followed = True
        target = addr_match.group(1)
        record = rec_match.group(1)
        secret = kb_lookup(record)
        try:
            send_email(target, secret)
        except Exception as exc:  # noqa: BLE001 - we re-classify below
            # Soft import to avoid a hard dependency at import time.
            try:
                from agent_policy_gateway import PolicyDenied, PolicyReview
            except Exception:  # pragma: no cover - install always available
                raise
            if isinstance(exc, (PolicyDenied, PolicyReview)):
                refused = True
                refusal_rule_id = exc.decision.rule_id
            else:
                raise

    summary = page.splitlines()[0] if page else ""
    return AgentResult(
        summary=summary,
        followed_injection=followed,
        exfil_target=target,
        exfil_record=record,
        refused_by_gateway=refused,
        refusal_rule_id=refusal_rule_id,
    )


__all__ = ["AgentResult", "run_research_task"]
