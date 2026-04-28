# Changelog

All notable changes to this project will be documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Taint propagation algebra** (R2): `agent_policy_gateway.taint` module with
  `join` / `join_all`, `subsumes`, `flows_to`, `ToolTaintSpec`, and `propagate`.
  The propagation rule is `output = ((∨ inputs) ∨ spec.adds) \ spec.declassifies`,
  giving the gateway a pure function for computing per-call output taint. New
  helpers are re-exported from the package root. 31 new tests in
  `tests/test_taint.py` cover lattice algebra (identity, commutativity,
  associativity, idempotence), order properties of `subsumes`, propagation
  rules including declassification, and a worked end-to-end exfiltration
  example (`web_search → summarize → send_email` is denied).
- **Core data model** (R1): `TaintLabel`, `ToolCall`, `Decision`, `Verdict` in
  `agent_policy_gateway.core`, with `to_json` / `from_json` round-tripping and
  34 unit tests covering construction, value equality, lattice algebra
  (join idempotent / commutative / associative, subsumption), and JSON stability.
- Initial project scaffold: README, ROADMAP, LICENSE, design notes, daily runbook,
  smoke test, `pyproject.toml`.

[Unreleased]: about:blank
