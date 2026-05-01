# Changelog

All notable changes to this project will be documented in this file. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **Append-only JSONL audit log + replay tool (R5)**: new `agent_policy_gateway.audit` module with `JsonlAuditWriter` (a callable compatible with the gateway's `AuditWriter` slot, opening the target file in append mode, flushing after every record, optional `fsync=True` for durability), `AuditRecord` (frozen `ts` / `call` / `decision` round-tripping through `to_dict` / `from_dict`), and `read_audit` (eager open, blank-line tolerant, raises `AuditFormatError` with line numbers on malformed JSON or missing keys). New `apg-replay` console script (entry point in `pyproject.toml`) reads a log and prints a human-readable timeline; supports `--verdict {allow,deny,review}` and `--limit N` filters and returns 0 / 2 / 3 for ok / missing-file / malformed-line. 26 new tests in `tests/test_audit.py` cover writer append semantics across reopens, parent-directory creation, single-line invariant under multi-line args, gateway integration on allow + deny paths, end-to-end exfiltration round-trip through the replay CLI, and the CLI's filter and exit-code behaviour. Public exports: `JsonlAuditWriter`, `AuditRecord`, `AuditFormatError`, `read_audit`, `replay_main`, `format_record`.
- **Reference monitor (R4)**: new `agent_policy_gateway.gateway` module with the `Gateway` class and `@gw.wrap_tool` decorator. The gateway evaluates policies in order (first match within a policy and across policies), propagates taint via `taint.propagate` to compute the output label on every decision, and writes an audit record before the wrapped tool runs (fail-closed-on-audit). Refusals raise `PolicyDenied` or `PolicyReview`, both carrying the originating `ToolCall` and `Decision`. The decorator recognises four reserved kwargs (`apg_input_label`, `apg_agent_id`, `apg_call_id`, `apg_resource`) and strips them before forwarding; `resource_arg` ties policy resource matching to a named parameter of the wrapped function via `inspect.signature`. `Action.RATE_LIMIT` maps to `Verdict.ALLOW` until the counter lands with R5. 27 new tests in `tests/test_gateway.py`, including a worked web→summarize→email exfiltration that the gateway denies at send. New public exports: `Gateway`, `GatewayError`, `PolicyDenied`, `PolicyReview`, `AuditWriter`, and the four reserved-kwarg constants.
- **Policy DSL v0 (R3)**: declarative YAML policies via the new
  `agent_policy_gateway.policy` module. Frozen Pydantic models —
  `Policy`, `Rule`, `Selector`, `Effect`, `TaintCondition`, `Action` —
  define a versioned schema with selectors over tool name (fnmatch
  glob), identity, resource glob, and three-clause taint conditions
  (`all_of` / `any_of` / `none_of`), and effects covering
  `allow` / `deny` / `review` / `rate_limit`. `load_policy`,
  `load_policy_str`, and `load_policies` parse and validate YAML,
  surfacing structural problems as `PolicyError`. Three example
  policies ship under `policies/`: `default.yaml` (baseline
  exfiltration prevention), `research-agent.yaml` (identity-gated
  publish, rate-limited search, reviewed external POST), and
  `strict-pii.yaml` (PII redaction-aware deny rules). 44 new tests in
  `tests/test_policy.py` cover schema validation, loader error paths,
  and first-match behaviour against the shipped examples.
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
