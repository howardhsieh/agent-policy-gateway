"""Tests for the R11 documentation site (mkdocs-material).

These tests validate that ``mkdocs.yml`` exists, parses, and points only at
files that actually live under ``docs/``. They also try to build the site if
``mkdocs`` is importable; otherwise that build assertion skips so the core
test suite stays dependency-light.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _load_mkdocs_yaml() -> dict:
    """Parse mkdocs.yml without instantiating a real ``mkdocs.config``.

    mkdocs uses ``!!python/name:...`` tags in some configs, but ours does
    not — a plain ``yaml.safe_load`` is sufficient and avoids requiring
    mkdocs at import time.
    """
    assert MKDOCS_YML.is_file(), f"missing {MKDOCS_YML}"
    with MKDOCS_YML.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_nav_files(nav: object) -> list[str]:
    """Walk an mkdocs nav structure and return every leaf file reference.

    nav is a list of either ``str`` (just a filename) or single-key dicts
    whose value is another nav node (``str``, ``list``, or external URL).
    External URLs (containing ``://``) are filtered out.
    """
    files: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, str):
            if "://" not in node:
                files.append(node)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if isinstance(node, dict):
            assert len(node) == 1, f"nav dict must have a single key: {node!r}"
            (_label, child), = node.items()
            visit(child)
            return
        raise TypeError(f"unsupported nav node type: {type(node).__name__}")

    visit(nav)
    return files


# --------------------------------------------------------------------------- #
# Structural invariants                                                       #
# --------------------------------------------------------------------------- #


class TestMkdocsConfig:
    def test_yaml_parses(self) -> None:
        config = _load_mkdocs_yaml()
        assert isinstance(config, dict)

    def test_required_top_level_keys(self) -> None:
        config = _load_mkdocs_yaml()
        for key in ("site_name", "theme", "nav"):
            assert key in config, f"mkdocs.yml missing required key: {key}"

    def test_site_name_set(self) -> None:
        config = _load_mkdocs_yaml()
        assert config["site_name"], "mkdocs.yml site_name must be non-empty"
        assert "agent-policy-gateway" in config["site_name"]

    def test_theme_is_material(self) -> None:
        config = _load_mkdocs_yaml()
        theme = config["theme"]
        # The theme can be either a bare string or a dict with `name`.
        if isinstance(theme, str):
            name = theme
        elif isinstance(theme, dict):
            name = theme.get("name")
        else:
            pytest.fail(f"theme must be str or mapping, got {type(theme).__name__}")
        assert name == "material", "R11 acceptance: theme must be mkdocs-material"

    def test_docs_dir_exists(self) -> None:
        config = _load_mkdocs_yaml()
        docs_dir = REPO_ROOT / config.get("docs_dir", "docs")
        assert docs_dir.is_dir(), f"docs_dir does not exist: {docs_dir}"

    def test_repo_url_points_at_github(self) -> None:
        config = _load_mkdocs_yaml()
        url = config.get("repo_url", "")
        assert url.startswith("https://github.com/"), (
            f"repo_url should be a github.com URL, got {url!r}"
        )

    def test_index_page_present(self) -> None:
        index_md = DOCS_DIR / "index.md"
        assert index_md.is_file(), "docs/index.md (homepage) must exist"

    def test_quickstart_page_present(self) -> None:
        quickstart = DOCS_DIR / "quickstart.md"
        assert quickstart.is_file(), "docs/quickstart.md must exist"


# --------------------------------------------------------------------------- #
# Nav <-> filesystem coherence                                                #
# --------------------------------------------------------------------------- #


class TestNavFilesResolve:
    def test_every_nav_file_exists(self) -> None:
        config = _load_mkdocs_yaml()
        files = _iter_nav_files(config["nav"])
        assert files, "mkdocs.yml nav must reference at least one file"
        missing = [f for f in files if not (DOCS_DIR / f).is_file()]
        assert not missing, f"nav references missing files: {missing}"

    def test_index_in_nav(self) -> None:
        config = _load_mkdocs_yaml()
        files = _iter_nav_files(config["nav"])
        assert "index.md" in files, "nav should expose the homepage as index.md"


# --------------------------------------------------------------------------- #
# Optional integration build (skipped when mkdocs is not installed)           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    importlib.util.find_spec("mkdocs") is None
    or importlib.util.find_spec("material") is None,
    reason=(
        "mkdocs/mkdocs-material not installed; "
        "install with `pip install -e .[docs]` to enable"
    ),
)
def test_mkdocs_build_clean(tmp_path: Path) -> None:
    """If mkdocs is installed, the site must build with --strict."""
    out_dir = tmp_path / "site"
    cmd = [
        sys.executable, "-m", "mkdocs", "build",
        "--strict",
        "--config-file", str(MKDOCS_YML),
        "--site-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"mkdocs build --strict failed:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    # Confirm a few expected output files landed.
    assert (out_dir / "index.html").is_file()
    assert (out_dir / "quickstart" / "index.html").is_file()
    # Clean up the rendered site eagerly to keep tmp small.
    shutil.rmtree(out_dir, ignore_errors=True)
