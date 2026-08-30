"""Worked example for the Progent symbolic-rule import PoC (R54).

``rules.json`` is a small banking-flavored policy in Progent's own
runtime format — ``{tool: [[priority, effect, condition, fallback],
...]}`` (see :mod:`agent_policy_gateway.progent_import`):

* ``get_balance`` — one unconditional allow rule.
* ``send_money`` — a priority-0 forbid on US recipients (bare-string
  regex, Progent's ``re.match`` shorthand), then a priority-1 allow
  restricted to one UK IBAN (JSON Schema ``pattern``) and two amounts
  (``enum``); everything else about the tool falls off the end and is
  denied.
* ``schedule_transaction`` — an allow restricted to ``recurring:
  false`` whose fallback ``2`` means Progent would ask the user; the
  translated policy routes those calls to ``review``.

Any tool without an entry (``read_file`` below) is denied by the
trailing catch-all, exactly as Progent denies tools without a policy.

:func:`run_demo` converts the file with :func:`convert_progent_policy`
and gates six representative calls through a real :class:`Gateway`,
returning ``(description, expected verdict, actual decision)`` triples
the tests and the ``__main__`` entry point both check.
"""

from __future__ import annotations

from pathlib import Path

from agent_policy_gateway import (
    Decision,
    Gateway,
    Policy,
    ToolCall,
    Verdict,
    convert_progent_policy,
    load_progent_policy,
)

#: Path to the Progent-format demo rules.
RULES_PATH = Path(__file__).resolve().parent / "rules.json"

#: The one UK recipient the send_money allow rule admits.
ALLOWED_RECIPIENT = "UK12345678901234567890"

#: (description, tool, args, expected verdict)
DEMO_CALLS: tuple[tuple[str, str, dict, Verdict], ...] = (
    ("balance check", "get_balance", {}, Verdict.ALLOW),
    (
        "transfer to the allowed UK IBAN",
        "send_money",
        {"recipient": ALLOWED_RECIPIENT, "amount": 100},
        Verdict.ALLOW,
    ),
    (
        "transfer of an unlisted amount",
        "send_money",
        {"recipient": ALLOWED_RECIPIENT, "amount": 9999},
        Verdict.DENY,
    ),
    (
        "transfer to a US recipient (forbid rule)",
        "send_money",
        {"recipient": "US99999999999999999999", "amount": 10},
        Verdict.DENY,
    ),
    (
        "recurring schedule (Progent would ask the user)",
        "schedule_transaction",
        {"recurring": True},
        Verdict.REVIEW,
    ),
    (
        "tool with no Progent entry",
        "read_file",
        {"path": "/etc/passwd"},
        Verdict.DENY,
    ),
)


def build_policy() -> Policy:
    """Convert ``rules.json`` into an APG policy."""
    return convert_progent_policy(
        load_progent_policy(RULES_PATH), name="progent-banking-demo"
    )


def build_gateway() -> Gateway:
    """A gateway enforcing only the converted Progent policy.

    ``default_deny`` stays ``False`` on purpose: the converted policy
    carries Progent's default-deny itself (the trailing catch-all rule).
    """
    return Gateway(policies=[build_policy()])


def run_demo() -> list[tuple[str, Verdict, Decision]]:
    """Gate the demo calls; returns (description, expected, decision)."""
    gateway = build_gateway()
    results: list[tuple[str, Verdict, Decision]] = []
    for description, tool, args, expected in DEMO_CALLS:
        decision = gateway.decide(ToolCall(tool_name=tool, args=args))
        results.append((description, expected, decision))
    return results


__all__ = [
    "ALLOWED_RECIPIENT",
    "DEMO_CALLS",
    "RULES_PATH",
    "build_gateway",
    "build_policy",
    "run_demo",
]
