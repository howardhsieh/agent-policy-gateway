"""Tests for the R22 PEP 561 ``py.typed`` marker.

The package is fully type-annotated, but those annotations are only visible to
a downstream type-checker (mypy / pyright) if the distribution ships a
``py.typed`` marker (PEP 561) *and* that marker is included as package data so
it lands in the built wheel/sdist. These tests lock in both halves:

* the marker exists on disk inside the importable package, and
* it is declared as setuptools package data in ``pyproject.toml``.

A final, guarded test actually builds a wheel and asserts ``py.typed`` is
inside it. That build is skipped when ``build`` is not importable, mirroring
``test_docs_site``'s skip pattern so the core suite stays dependency-light.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

try:  # tomllib is stdlib on 3.11+; fall back to tomli if present
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised only on 3.10
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover - no TOML parser available
        tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "src" / "agent_policy_gateway"
MARKER = PKG_DIR / "py.typed"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_marker_file_exists() -> None:
    assert MARKER.is_file(), f"missing PEP 561 marker {MARKER}"


def test_marker_lives_inside_the_importable_package() -> None:
    """The marker must sit next to ``__init__.py`` to be honoured by PEP 561."""
    assert (PKG_DIR / "__init__.py").is_file()
    assert MARKER.parent == PKG_DIR


def test_marker_is_empty_or_partial_form() -> None:
    """A standard ``py.typed`` is empty (whole package typed). The string
    ``partial`` is the only other PEP 561-legal content; reject anything else."""
    content = MARKER.read_text(encoding="utf-8").strip()
    assert content in ("", "partial"), f"unexpected py.typed content: {content!r}"


@pytest.mark.skipif(
    tomllib is None,
    reason="no TOML parser available (Python 3.10 without 'tomli')",
)
def test_pyproject_declares_marker_as_package_data() -> None:
    cfg = _pyproject()
    pkg_data = (
        cfg.get("tool", {})
        .get("setuptools", {})
        .get("package-data", {})
    )
    patterns = pkg_data.get("agent_policy_gateway", [])
    assert "py.typed" in patterns, (
        "pyproject [tool.setuptools.package-data] must ship py.typed so it "
        f"lands in the wheel; got {pkg_data!r}"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("build") is None,
    reason="the 'build' package is not installed; skipping wheel-build assertion",
)
def test_marker_is_included_in_built_wheel(tmp_path: Path) -> None:
    """End-to-end: build a wheel and assert py.typed is inside it."""
    out = tmp_path / "dist"
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(REPO_ROOT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"wheel build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = list(out.glob("*.whl"))
    assert wheels, "no wheel produced"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert "agent_policy_gateway/py.typed" in names, (
        f"py.typed missing from wheel; members: {names}"
    )
