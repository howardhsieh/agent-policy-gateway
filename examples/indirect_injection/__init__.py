"""Worked example: indirect prompt injection in a web-research agent (R10).

Scenario
--------
A research agent is given the task "summarise the latest news on widget
manufacturing". It uses a ``web_fetch`` tool to read a page from the open web,
then has access to ``kb_lookup`` (an internal knowledge base) and
``send_email`` (an outbound message tool).

The page it fetches has been *poisoned*: alongside its visible content it
contains a hidden instruction telling the agent to look up an internal record
and email it to ``attacker@evil.example``. This is the canonical *indirect
prompt injection* attack — the malicious text never reaches the agent through
its operator, it arrives as a side effect of the agent doing its job.

The example ships two runners:

* :mod:`examples.indirect_injection.naive` — the agent has direct access to
  the tool functions. The injected instruction is followed; ``send_email`` is
  invoked; an email is delivered to the attacker.
* :mod:`examples.indirect_injection.gated` — the same agent and the same
  injected page, but every tool call is mediated by an
  :class:`agent_policy_gateway.Gateway` configured with
  ``policies/default.yaml``. The ``deny-web-to-email`` rule sees that
  ``send_email`` is being called with a label carrying the ``web`` source and
  refuses the call before the email leaves the system.

Reproduce from a single command::

    python -m examples.indirect_injection

This prints the outcome of both runners and exits 0. The acceptance criterion
for R10 is that running this command shows the naive agent exfiltrating the
secret and the gated agent refusing it.

The "LLM" used here is intentionally a deterministic stub that follows
instructions found in fetched page content. The point of the example is to
demonstrate the *defence*, not to evaluate any real model. The defence works
identically against any agent — including an actual LLM-driven one — because
the gateway makes its decision from the taint label on the call's inputs, not
from anything the agent says or thinks.
"""

from examples.indirect_injection.gated import run_gated
from examples.indirect_injection.injected_page import (
    ATTACKER_EMAIL,
    INJECTED_PAGE_HTML,
    INJECTED_PAGE_URL,
)
from examples.indirect_injection.naive import run_naive

__all__ = [
    "ATTACKER_EMAIL",
    "INJECTED_PAGE_HTML",
    "INJECTED_PAGE_URL",
    "run_gated",
    "run_naive",
]
