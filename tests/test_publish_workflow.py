"""Tests for the R14b PyPI publish workflow file.

These tests lock in the *structural* invariants of
``.github/workflows/publish.yml`` so a careless edit can't silently
break the release pipeline. They deliberately do not assert against
specific action versions or step ordering beyond what the runbook
commits to — the goal is to make sure that a release tag still
triggers a ``test -> build -> publish`` pipeline that publishes via
PyPI trusted publishing into the ``pypi`` environment.

Companion docs:
- ``docs/release.md`` (R14a) — the release runbook.
- ``docs/design.md`` — the R14a/R14b split rationale.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_FILE = REPO_ROOT / ".github" / "workflows" / "publish.yml"
RELEASE_MD = REPO_ROOT / "docs" / "release.md"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _load_workflow() -> dict:
    with WORKFLOW_FILE.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert isinstance(cfg, dict), "workflow YAML must parse to a mapping"
    return cfg


def _triggers(cfg: dict):
    """Return the ``on:`` section.

    PyYAML's ``safe_load`` resolves the bare key ``on`` to the YAML 1.1
    boolean ``True``, so we look up both forms before giving up.
    """
    if "on" in cfg:
        return cfg["on"]
    if True in cfg:
        return cfg[True]
    raise AssertionError("workflow has no ``on:`` trigger section")


def _needs(job: dict) -> list[str]:
    needs = job.get("needs")
    if needs is None:
        return []
    if isinstance(needs, str):
        return [needs]
    if isinstance(needs, list):
        return [str(n) for n in needs]
    raise AssertionError(f"unexpected ``needs`` shape: {needs!r}")


def _step_text(step: dict) -> str:
    """Concatenate the user-visible fields of a step for substring checks."""
    fields = (step.get("name") or "", step.get("run") or "", step.get("uses") or "")
    return " \n ".join(f for f in fields if f)


# --------------------------------------------------------------------------- #
# File presence and shape                                                     #
# --------------------------------------------------------------------------- #


def test_workflow_file_exists():
    assert WORKFLOW_FILE.is_file(), f"missing {WORKFLOW_FILE}"


def test_workflow_file_is_non_trivial():
    assert WORKFLOW_FILE.stat().st_size > 500, "publish workflow looks empty"


def test_workflow_file_parses_as_yaml():
    cfg = _load_workflow()
    assert isinstance(cfg, dict)


def test_workflow_has_human_readable_name():
    cfg = _load_workflow()
    name = cfg.get("name")
    assert isinstance(name, str) and name.strip(), (
        "workflow must declare a non-empty top-level ``name``"
    )


# --------------------------------------------------------------------------- #
# Triggers                                                                    #
# --------------------------------------------------------------------------- #


def test_workflow_triggers_on_v_star_tag_push():
    cfg = _load_workflow()
    on = _triggers(cfg)
    assert isinstance(on, dict), "``on:`` must be a mapping of triggers"
    push = on.get("push")
    assert isinstance(push, dict), "workflow must trigger on push"
    tags = push.get("tags")
    assert isinstance(tags, list) and tags, "push trigger must specify tags"
    assert "v*" in tags, f"tag filter must include 'v*'; got {tags!r}"


def test_workflow_supports_workflow_dispatch():
    cfg = _load_workflow()
    on = _triggers(cfg)
    assert "workflow_dispatch" in on, (
        "workflow must be runnable manually via ``workflow_dispatch``"
    )


# --------------------------------------------------------------------------- #
# Job graph                                                                   #
# --------------------------------------------------------------------------- #


def test_workflow_has_exactly_three_named_jobs():
    cfg = _load_workflow()
    jobs = cfg.get("jobs")
    assert isinstance(jobs, dict), "workflow must declare a ``jobs:`` mapping"
    assert set(jobs.keys()) == {"test", "build", "publish"}, (
        f"expected exactly ``test``, ``build``, ``publish`` jobs, "
        f"got {sorted(jobs.keys())}"
    )


def test_build_depends_on_test():
    cfg = _load_workflow()
    assert "test" in _needs(cfg["jobs"]["build"]), (
        "``build`` job must declare ``needs: [test]``"
    )


def test_publish_depends_on_build():
    cfg = _load_workflow()
    assert "build" in _needs(cfg["jobs"]["publish"]), (
        "``publish`` job must declare ``needs: [build]``"
    )


def test_jobs_run_on_ubuntu():
    cfg = _load_workflow()
    for name, job in cfg["jobs"].items():
        runs_on = job.get("runs-on")
        assert isinstance(runs_on, str) and "ubuntu" in runs_on.lower(), (
            f"job {name!r} should run on ubuntu, got {runs_on!r}"
        )


# --------------------------------------------------------------------------- #
# Per-job step invariants                                                     #
# --------------------------------------------------------------------------- #


def test_test_job_runs_pytest_and_ruff():
    cfg = _load_workflow()
    steps = cfg["jobs"]["test"].get("steps", [])
    text = " \n ".join(_step_text(s) for s in steps if isinstance(s, dict))
    assert "pytest" in text, "test job must run pytest"
    assert "ruff" in text, "test job must run ruff"


def test_build_job_runs_python_build_and_twine_check():
    cfg = _load_workflow()
    steps = cfg["jobs"]["build"].get("steps", [])
    text = " \n ".join(_step_text(s) for s in steps if isinstance(s, dict))
    assert "python -m build" in text, "build job must run ``python -m build``"
    assert "twine check" in text, "build job must run ``twine check`` on the artifacts"


def test_build_job_uploads_distributions_artifact():
    cfg = _load_workflow()
    steps = cfg["jobs"]["build"].get("steps", [])
    uses = [s.get("uses", "") for s in steps if isinstance(s, dict)]
    assert any(u.startswith("actions/upload-artifact") for u in uses), (
        "build job must upload the produced distributions as an artifact "
        "so the publish job can consume them"
    )


def test_publish_job_downloads_distributions_artifact():
    cfg = _load_workflow()
    steps = cfg["jobs"]["publish"].get("steps", [])
    uses = [s.get("uses", "") for s in steps if isinstance(s, dict)]
    assert any(u.startswith("actions/download-artifact") for u in uses), (
        "publish job must download the build artifacts before uploading"
    )


def test_publish_job_uses_pypa_publish_action():
    cfg = _load_workflow()
    steps = cfg["jobs"]["publish"].get("steps", [])
    uses = [s.get("uses", "") for s in steps if isinstance(s, dict)]
    assert any(u.startswith("pypa/gh-action-pypi-publish") for u in uses), (
        "publish job must use ``pypa/gh-action-pypi-publish``"
    )


# --------------------------------------------------------------------------- #
# Trusted publishing invariants                                               #
# --------------------------------------------------------------------------- #


def test_publish_job_requests_id_token_write_permission():
    cfg = _load_workflow()
    publish = cfg["jobs"]["publish"]
    perms = publish.get("permissions")
    assert isinstance(perms, dict), (
        "publish job must declare a ``permissions`` block for OIDC"
    )
    assert perms.get("id-token") == "write", (
        "publish job must request ``id-token: write`` for trusted publishing; "
        f"got {perms!r}"
    )


def test_publish_job_uses_pypi_environment():
    cfg = _load_workflow()
    publish = cfg["jobs"]["publish"]
    env = publish.get("environment")
    if isinstance(env, str):
        name = env
    elif isinstance(env, dict):
        name = env.get("name")
    else:
        name = None
    assert name == "pypi", (
        f"publish job must use the ``pypi`` GitHub environment; got {env!r}"
    )


def test_no_pypi_api_token_secret_referenced():
    text = WORKFLOW_FILE.read_text(encoding="utf-8")
    assert "PYPI_API_TOKEN" not in text, (
        "publish workflow must not reference a stored ``PYPI_API_TOKEN`` "
        "secret — trusted publishing is the destination"
    )


def test_workflow_environment_matches_release_runbook():
    runbook = RELEASE_MD.read_text(encoding="utf-8").lower()
    assert "pypi" in runbook, "release runbook must name the pypi environment"
    text = WORKFLOW_FILE.read_text(encoding="utf-8")
    assert "name: pypi" in text, (
        "workflow ``environment.name`` must match the release runbook (``pypi``)"
    )


# --------------------------------------------------------------------------- #
# End-to-end acceptance: python -m build produces a twine-clean artifact set  #
# --------------------------------------------------------------------------- #


def _have_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not (_have_module("build") and _have_module("twine")),
    reason="``build`` and/or ``twine`` not installed; "
    "skip the end-to-end packaging check on lean test matrices",
)
def test_python_m_build_produces_twine_clean_artifacts(tmp_path):
    """Acceptance criterion for R14b: ``python -m build`` produces a
    twine-clean sdist + wheel from the project root.
    """
    out = tmp_path / "dist"
    build_result = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out), str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert build_result.returncode == 0, (
        f"``python -m build`` failed (exit {build_result.returncode}):\n"
        f"--- stdout ---\n{build_result.stdout}\n"
        f"--- stderr ---\n{build_result.stderr}"
    )
    artifacts = sorted(out.iterdir())
    sdists = [p for p in artifacts if p.suffix == ".gz"]
    wheels = [p for p in artifacts if p.suffix == ".whl"]
    assert sdists, f"no sdist produced; got {[p.name for p in artifacts]}"
    assert wheels, f"no wheel produced; got {[p.name for p in artifacts]}"

    twine_result = subprocess.run(
        [sys.executable, "-m", "twine", "check", "--strict", *map(str, artifacts)],
        capture_output=True,
        text=True,
    )
    assert twine_result.returncode == 0, (
        f"``twine check --strict`` failed (exit {twine_result.returncode}):\n"
        f"--- stdout ---\n{twine_result.stdout}\n"
        f"--- stderr ---\n{twine_result.stderr}"
    )
    stray_build = REPO_ROOT / "build"
    if stray_build.is_dir():
        shutil.rmtree(stray_build, ignore_errors=True)
