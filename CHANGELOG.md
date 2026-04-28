# Changelog

All notable changes to this project will be documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Core data model** (R1): `TaintLabel`, `ToolCall`, `Decision`, `Verdict` in
  `agent_policy_gateway.core`, with `to_json` / `from_json` round-tripping and
  34 unit tests covering construction, value equality, lattice algebra
  (join idempotent / commutative / associative, subsumption), and JSON stability.
- Initial project scaffold: README, ROADMAP, LICENSE, design notes, daily runbook,
  smoke test, `pyproject.toml`.

[Unreleased]: about:blank
