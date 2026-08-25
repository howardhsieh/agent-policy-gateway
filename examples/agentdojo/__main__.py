"""Entry point for the AgentDojo banking-suite demo.

Run from the repository root (needs the ``agentdojo`` extra)::

    python -m examples.agentdojo

Runs the benign episode (must pass clean) and the injection episode (the
attacker-directed ``send_money`` must be refused, and no money moved).
Exits 0 on the expected outcome, 1 otherwise, 3 when ``agentdojo`` is not
installed — so the entry point doubles as a CI sanity check.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import agentdojo  # noqa: F401
    except ImportError:
        print(
            "examples.agentdojo requires the AgentDojo benchmark: "
            "pip install agentdojo",
            file=sys.stderr,
        )
        return 3

    from examples.agentdojo import (
        attacker_was_paid,
        run_benign,
        run_injection,
    )

    code = 0

    benign = run_benign()
    if benign.errors:
        print(f"benign: REFUSED unexpectedly: {benign.errors}")
        code = 1
    else:
        print(f"benign: all {len(benign.steps)} calls allowed (read-only task passes)")

    injection = run_injection()
    read_error, sink_error = injection.steps[0][1], injection.steps[1][1]
    if read_error is not None:
        print(f"injection: read unexpectedly refused: {read_error}")
        code = 1
    elif sink_error is None or not sink_error.startswith("PolicyDenied"):
        print(f"injection: exfiltration NOT blocked (error={sink_error!r})")
        code = 1
    elif attacker_was_paid(injection):
        print("injection: refusal reported but money moved (BUG)")
        code = 1
    else:
        print(f"injection: exfiltration blocked — {sink_error}")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
