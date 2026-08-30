"""Entry point for the Progent-import demo (R54).

Run from the repository root::

    python -m examples.progent

Converts ``rules.json`` (Progent's native format) into an APG policy,
prints the generated YAML, gates six representative calls, and exits 0
only when every decision matches the Progent semantics — so the entry
point doubles as a CI sanity check.
"""

from __future__ import annotations

import sys

from agent_policy_gateway import policy_to_yaml
from examples.progent import build_policy, run_demo


def main() -> int:
    print("=== converted policy ===")
    print(policy_to_yaml(build_policy()))
    print("=== decisions ===")
    code = 0
    for description, expected, decision in run_demo():
        ok = decision.verdict == expected
        status = "ok " if ok else "FAIL"
        print(
            f"[{status}] {description}: {decision.verdict.value} "
            f"(rule {decision.rule_id})"
        )
        if not ok:
            print(f"       expected {expected.value}")
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
