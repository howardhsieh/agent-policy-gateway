"""Entry point for the long-horizon stateful eval demo (R55).

Run from the repository root::

    python -m examples.stateful

Replays the laundering-adversary scenario family under all three arms,
prints the summary table, then checks the R55 invariants and exits 0 only
when every one holds — so the entry point doubles as a CI sanity check.
"""

from __future__ import annotations

import sys

from agent_policy_gateway.stateful_benchmark import render_demo_table, run_demo
from examples.stateful import expectations_hold


def main() -> int:
    summaries = run_demo()
    print("=== stateful long-horizon benchmark (R55) ===")
    print(render_demo_table(summaries))
    print("=== invariants ===")
    code = 0
    for claim, ok in expectations_hold({s["arm"]: s for s in summaries}):
        status = "ok  " if ok else "FAIL"
        print(f"[{status}] {claim}")
        if not ok:
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
