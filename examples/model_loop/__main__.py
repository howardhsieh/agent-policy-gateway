"""Entry point for the model-in-the-loop benchmark demo (R59).

Run from the repository root (needs the ``agentdojo`` extra, no API key)::

    python -m examples.model_loop

Replays the committed driver fixtures through the runner, prints the
benchmark table, then checks the R59 invariants and exits 0 only when
every one holds — so the entry point doubles as a CI sanity check.
"""

from __future__ import annotations

import sys

from agent_policy_gateway.model_benchmark import render_table
from examples.model_loop import expectations_hold, run_replay


def main() -> int:
    rows = run_replay()
    print("=== model-in-the-loop benchmark (R59, replayed fixtures) ===")
    print(render_table(rows))
    print("=== invariants ===")
    code = 0
    for claim, ok in expectations_hold(rows):
        status = "ok  " if ok else "FAIL"
        print(f"[{status}] {claim}")
        if not ok:
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
