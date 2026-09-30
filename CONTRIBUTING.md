# Contributing

Thanks for helping. APG is built incrementally and in public: see
[`ROADMAP.md`](ROADMAP.md) for what is planned and
[`docs/design.md`](docs/design.md) for the architecture.

## Setup

```bash
git clone https://github.com/howardhsieh/agent-policy-gateway.git
cd agent-policy-gateway
pip install -e ".[dev,docs]"
```

## Before opening a pull request

Run the same gates CI runs:

```bash
ruff check .
mypy src/agent_policy_gateway
pytest
mkdocs build --strict
```

- Add tests for every behavior change. Security decisions (allow, deny,
  review, redact, taint propagation, declassification) need a test for the
  case that must be blocked, not only the case that must pass.
- Keep `docs/cli.md` in step with the CLI; a test checks every flag.
- Benchmark numbers in the docs are pinned by tests; regenerate them with the
  command named next to each table rather than editing numbers by hand.
- Update `CHANGELOG.md` under `[Unreleased]`.

## Reporting bugs and bypasses

Bugs: open an issue. Policy bypasses and other vulnerabilities: report them
privately, see [SECURITY.md](SECURITY.md).

By contributing you agree that your contribution is licensed under the
project's [Apache-2.0 license](LICENSE) and that you follow the
[code of conduct](CODE_OF_CONDUCT.md).
