"""Entry point for regenerating the committed TraceSig example traces (R62).

Run from the repository root::

    python -m examples.traces

Rewrites ``<scenario>.audit.jsonl`` and ``<scenario>.tracesig.jsonl`` for
all three scenarios into ``examples/traces/``, prints the narrative
invariant checks, and exits 0 only when every one holds — so the entry
point doubles as a CI sanity check.
"""

from __future__ import annotations

import sys

from examples.traces import expectations_hold, generate_traces


def main() -> int:
    generated = generate_traces()
    print("=== TraceSig example traces (R62) ===")
    for name, (audit_path, tracesig_path) in generated.items():
        print(f"{name}:")
        print(f"  audit:    {audit_path}")
        print(f"  tracesig: {tracesig_path}")
    print("=== invariants ===")
    code = 0
    for claim, ok in expectations_hold(generated):
        status = "ok  " if ok else "FAIL"
        print(f"[{status}] {claim}")
        if not ok:
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
