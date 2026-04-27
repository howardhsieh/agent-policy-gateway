# Roadmap

The daily scheduled task reads this file, picks the **first unchecked item under "Up next"**,
implements it, runs tests, commits, and pushes. When an item is done, it is moved to "Done"
with the date and commit hash.

Keep items small enough to finish in a single run (roughly 1–3 hours of focused work).
Each item must include acceptance criteria so the daily task knows when it's done.

---

## Up next

- [ ] **R1. Core data model: `ToolCall`, `TaintLabel`, `Decision`.**
  - `src/agent_policy_gateway/core.py` defines dataclasses for a tool call, the labels
    attached to its inputs/outputs, and the gateway's decision (`ALLOW` / `DENY` / `REVIEW`).
  - Unit tests in `tests/test_core.py` cover construction, equality, and serialization
    (to/from JSON).
  - Done when: `pytest tests/test_core.py` passes and coverage on `core.py` is ≥90%.

- [ ] **R2. Taint propagation algebra.**
  - `src/agent_policy_gateway/taint.py` implements label join (`∨`), subsumption (`⊑`),
    and propagation rules: outputs are tainted with the join of all input taints plus the
    tool's own source labels.
  - Tests cover identity, commutativity, associativity, and a worked example
    (web_search → summarize → send_email refuses on high-web taint).

- [ ] **R3. Policy DSL v0 (YAML).**
  - Schema for declarative policies: rules with selectors (tool name, identity, resource
    pattern), conditions on taint, and effects (allow/deny/review/rate-limit).
  - Loader, validator (Pydantic), and 3 example policies under `policies/`.

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

_(Items move here with completion date and short commit reference.)_
