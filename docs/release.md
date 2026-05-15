# Release runbook

This page documents how a new version of `agent-policy-gateway` reaches
PyPI. There are two release paths:

- The **automated path** — pushing a `v*` tag runs a GitHub Actions
  workflow that builds the sdist + wheel and uploads them via PyPI
  trusted publishing. The workflow file at
  `.github/workflows/publish.yml` is the deliverable of roadmap
  item **R14b** and is not yet on `main`; until it ships, the
  manual fallback in §3 is the only path.
- The **manual fallback** — `python -m build` + `twine upload` from
  a developer machine. This works today and will remain the
  documented disaster-recovery path even after R14b lands.

The runbook has three parts:

1. **One-time setup** — configuring PyPI trusted publishing so the
   workflow can upload without a stored API token.
2. **Cutting a release (automated)** — the steady-state procedure
   once R14b ships the workflow file.
3. **Manual fallback** — building and uploading from a developer
   machine when the workflow can't (or doesn't yet) run.

## 1. One-time setup (trusted publishing)

The publish workflow uses [PyPI trusted publishing] (OpenID Connect)
rather than a stored API token. Configuring it is a one-time step on
the PyPI project page. You can do this before R14b lands — the
configuration is inert until the workflow file is present and a tag
is pushed.

[PyPI trusted publishing]: https://docs.pypi.org/trusted-publishers/

1. Create (or claim) the `agent-policy-gateway` project on PyPI:
   <https://pypi.org/manage/account/publishing/>.
2. Under *Add a new pending publisher*, fill in:
    - **PyPI Project Name**: `agent-policy-gateway`
    - **Owner**: `howardhsieh`
    - **Repository name**: `agent-policy-gateway`
    - **Workflow name**: `publish.yml`
    - **Environment name**: `pypi`
3. On the GitHub repo, create the `pypi` environment:
   *Settings → Environments → New environment → `pypi`*. Optional
   but recommended: add a required-reviewer protection rule so the
   publish job pauses until you approve it.

That's it — no `PYPI_API_TOKEN` secret is needed.

## 2. Cutting a release (automated, after R14b ships)

Once R14b has shipped the publish workflow and trusted publishing is
configured, the steady-state release flow is three steps:

1. Bump the version in `pyproject.toml` (e.g. `0.0.1` → `0.1.0`) and
   move the relevant `[Unreleased]` block in `CHANGELOG.md` under a
   new `[<version>] — YYYY-MM-DD` heading. Commit both on `main`.
2. Tag the release commit and push:
   ```bash
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin v0.1.0
   ```
3. Watch the *Publish to PyPI* workflow run under the GitHub Actions
   tab. The `test → build → publish` jobs run in sequence; if any
   step is red, the publish job will not start. If the `pypi`
   environment has a reviewer rule, approve the run when prompted.

The workflow will also have a `workflow_dispatch` trigger so it can
be run manually from the Actions tab — useful for re-running a
release after a transient failure.

## 3. Manual fallback

If GitHub Actions is unavailable, or while R14b is still pending,
the same artifacts can be built and uploaded from a developer
machine. This path uses an API token rather than trusted publishing,
so it should only be used when the workflow can't run.

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

`twine` will prompt for credentials; use a project-scoped PyPI API
token for the upload. Run `twine check` first so a malformed README
or missing metadata is caught before the upload attempt.

## Verification

After the workflow finishes (or after a manual upload), confirm the
release looks healthy:

- The project page at <https://pypi.org/project/agent-policy-gateway/>
  shows the new version with both an sdist and a wheel.
- `pip install agent-policy-gateway==<version>` succeeds in a clean
  virtualenv.
- `python -c "import agent_policy_gateway, sys; print(agent_policy_gateway.__name__)"`
  prints `agent_policy_gateway`.
- The console scripts work: `apg-replay --help` and
  `apg-bench --help` both exit 0.

## What lives where

- Package metadata that PyPI displays:
  [`pyproject.toml`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/pyproject.toml).
- The user-facing changelog:
  [`CHANGELOG.md`](https://github.com/howardhsieh/agent-policy-gateway/blob/main/CHANGELOG.md).
- Structural invariants for this runbook (file presence, mkdocs nav,
  README and index cross-links, internal-link integrity) are locked
  in by `tests/test_release_runbook.py`. The structural invariants
  for the publish workflow itself will land with R14b in
  `tests/test_publish_workflow.py`.
