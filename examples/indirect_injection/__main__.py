"""Entry point for the indirect-injection demo.

Run from the repository root::

    python -m examples.indirect_injection

This runs the naive variant first and prints what was exfiltrated, then the
gated variant and prints the rule that refused the call. Exits 0 on the
expected outcome (naive exfiltrates, gated refuses), 1 otherwise — that
makes the entry point usable as a sanity check in CI.
"""

from __future__ import annotations

import sys

from examples.indirect_injection.gated import run_gated
from examples.indirect_injection.naive import run_naive


def _summarise_naive() -> int:
    result = run_naive()
    sent = result.outbox.sent
    if not sent or not result.agent.followed_injection:
        print("naive: did NOT exfiltrate (unexpected for the demo)")
        return 1
    last = sent[-1]
    print(
        "naive: exfiltrated record "
        f"'{result.agent.exfil_record}' (body={last.body!r}) to {last.to}"
    )
    return 0


def _summarise_gated() -> int:
    result = run_gated()
    if result.outbox.sent:
        print("gated: email LEAKED (gateway failed to refuse)")
        return 1
    if not result.agent.refused_by_gateway:
        print("gated: agent did not see a refusal (unexpected)")
        return 1
    print(
        "gated: refused by rule "
        f"'{result.agent.refusal_rule_id}' — outbox empty, "
        f"{len(result.audit)} audit record(s) written"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run both variants and print a one-line outcome for each."""
    del argv  # unused; reserved for future flags
    rc_naive = _summarise_naive()
    rc_gated = _summarise_gated()
    return rc_naive | rc_gated


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess test
    sys.exit(main(sys.argv[1:]))
