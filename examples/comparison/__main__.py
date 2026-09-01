"""Entry point for the APG / Progent / Fides comparison demo (R56).

Run from the repository root::

    python -m examples.comparison

Replays the comparison scenario family under all six arms, prints the
per-variant matrices, then checks the R56 invariants and exits 0 only
when every one holds — so the entry point doubles as a CI sanity check.
"""

from __future__ import annotations

import sys

from agent_policy_gateway.comparison_benchmark import (
    render_comparison_table,
    run_comparison,
)
from examples.comparison import expectations_hold


def main() -> int:
    summaries = run_comparison()
    print("=== APG / Progent / Fides comparison benchmark (R56) ===")
    print(render_comparison_table(summaries))
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
