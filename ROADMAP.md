# Roadmap

The daily scheduled task reads this file, picks the **first unchecked item under "Up next"**,
implements it, runs tests, commits, and pushes. When an item is done, it is moved to "Done"
with the date and commit hash.

Keep items small enough to finish in a single run (roughly 1–3 hours of focused work).
Each item must include acceptance criteria so the daily task knows when it's done.

---

## Up next

- [ ] **R4. Reference `Gateway` class and `wrap_tool` decorator.**
  - Synchronous version. The decorator records the call, evaluates policies, propagates
    taint, and writes to the audit log.

- [ ] **R5. Append-only JSONL audit log + replay tool.**
  - `audit.py` writes one JSON object per decision. CLI `apg-replay` reads a log and
    prints a human-readable timeline.

- [ ] **R6. MCP adapter.**
  - Given an MCP server, expose its tools through the gateway with one line of code.

- [ ] **R7. OpenAI function-calling adapter.**

- [ ] **R8. Anthropic tool-use adapter.**

- [ ] **R9. Async gateway path.**

- [ ] **R10. Worked end-to-end example: indirect prompt injection in a web-research agent.**
  - `examples/indirect_injection/` shows a deliberately injected web page, a naive agent
    that falls for it, and the same agent gated by `agent-policy-gateway` refusing the
    exfiltration. Reproducible from a single command.

- [ ] **R11. Documentation site (mkdocs-material).**

- [ ] **R12. Benchmark harness: measure overhead per call, throughput.**

- [ ] **R13. Threat-model document with adversary classes and assumptions.**

- [ ] **R14. PyPI publish workflow.**

- [ ] **R15. v0.1.0 release.**

---

## Done

- **R1. Core data model** — completed 2026-04-28. Added `core.py` with `TaintLabel`, `ToolCall`, `Verdict`, `Decision`, plus `to_json` / `from_json`. 34 tests in `tests/test_core.py`, all green; ruff clean. Commit `0877c07`.

- **R2. Taint propagation algebra** — completed 2026-04-28. Added `taint.py` with `join` / `join_all`, `subsumes`, `flows_to`, `ToolTaintSpec`, and `propagate` (`output = ((∨ inputs) ∨ adds) \ declassifies`). 31 new tests in `tests/test_taint.py` covering lattice algebra, propagation rules, and a worked `web_search → summarize → send_email` exfiltration refusal. 65/65 tests green; ruff clean. Commit `fe45868`.

- **R3. Policy DSL v0 (YAML)** — completed 2026-04-29. Added `policy.py` with frozen Pydantic models `Policy` / `Rule` / `Selector` / `Effect` / `TaintCondition` / `Action`, plus `load_policy` / `load_policy_str` / `load_policies`. Selectors support `fnmatch`-globbed tool and resource names, identity equality, and three-clause taint conditions (`all_of` / `any_of` / `none_of`); effects validate that `rate_limit` requires a positive `limit_per_minute` and other actions reject one. Three example YAML policies under `policies/` (`default.yaml`, `research-agent.yaml`, `strict-pii.yaml`) demonstrate exfiltration prevention, identity-based publish gating, and PII handling. 44 new tests in `tests/test_policy.py` cover schema validation, loader error paths, and first-match behaviour against the example policies. 109/109 tests green; ruff clean. Commit `__HASH__`.

_(More items below as they ship.)_
