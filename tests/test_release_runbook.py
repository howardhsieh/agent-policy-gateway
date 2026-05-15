"""Tests for the R14a release runbook.

These tests lock in the structural invariants of ``docs/release.md``
and its cross-links into the rest of the documentation. They
intentionally do **not** check the publish workflow file
(``.github/workflows/publish.yml``); that file ships with R14b
together with ``tests/test_publish_workflow.py``.

The test scope mirrors the commit scope: we only assert what R14a
actually puts on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_MD = REPO_ROOT / "docs" / "release.md"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
INDEX_MD = REPO_ROOT / "docs" / "index.md"
README_MD = REPO_ROOT / "README.md"
DESIGN_MD = REPO_ROOT / "docs" / "design.md"


# --------------------------------------------------------------------------- #
# File presence and shape                                                     #
# --------------------------------------------------------------------------- #


def test_release_doc_exists():
    assert RELEASE_MD.is_file(), f"missing {RELEASE_MD}"


def test_release_doc_is_non_trivial():
    # Empty / placeholder runbooks are worse than no runbook.
    assert RELEASE_MD.stat().st_size > 500, "release runbook looks empty"


def test_release_doc_starts_with_atx_h1():
    first = RELEASE_MD.read_text(encoding="utf-8").lstrip().splitlines()[0]
    assert first.startswith("# "), "release.md should start with an ATX H1"


# --------------------------------------------------------------------------- #
# Required sections                                                           #
# --------------------------------------------------------------------------- #


REQUIRED_SECTIONS = (
    "## 1. One-time setup",
    "## 2. Cutting a release",
    "## 3. Manual fallback",
    "## Verification",
    "## What lives where",
)


@pytest.mark.parametrize("heading", REQUIRED_SECTIONS)
def test_release_doc_has_required_section(heading: str):
    text = RELEASE_MD.read_text(encoding="utf-8")
    assert heading in text, f"release.md is missing a section starting with {heading!r}"


def test_release_doc_sections_appear_in_order():
    text = RELEASE_MD.read_text(encoding="utf-8")
    positions = [text.index(h) for h in REQUIRED_SECTIONS]
    assert positions == sorted(positions), (
        f"release.md section order is wrong: got {positions}"
    )


# --------------------------------------------------------------------------- #
# Content invariants                                                          #
# --------------------------------------------------------------------------- #


def test_release_doc_describes_trusted_publishing():
    text = RELEASE_MD.read_text(encoding="utf-8")
    assert "trusted publishing" in text.lower(), (
        "release runbook must describe PyPI trusted publishing"
    )


def test_release_doc_documents_manual_fallback_commands():
    text = RELEASE_MD.read_text(encoding="utf-8")
    # The manual fallback must spell out the actual commands a
    # developer would run, not just describe them in prose.
    for snippet in ("python -m build", "python -m twine upload"):
        assert snippet in text, (
            f"release runbook must document the {snippet!r} command"
        )


def test_release_doc_documents_twine_check():
    # twine check catches malformed README / metadata before upload.
    text = RELEASE_MD.read_text(encoding="utf-8")
    assert "twine check" in text, (
        "manual fallback should run ``twine check`` before upload"
    )


def test_release_doc_documents_pypi_environment_name():
    text = RELEASE_MD.read_text(encoding="utf-8")
    # The PyPI trusted-publisher form requires an environment name;
    # the runbook should commit to ``pypi`` so the workflow that
    # eventually lands matches.
    assert "pypi" in text.lower()


def test_release_doc_acknowledges_r14b_split():
    # R14a ships without the workflow file. The runbook should be
    # honest about that — it points at .github/workflows/publish.yml
    # but explicitly notes the workflow file lands with R14b.
    text = RELEASE_MD.read_text(encoding="utf-8")
    assert "R14b" in text, (
        "release runbook should mention R14b so readers know the "
        "workflow file is forthcoming"
    )


def test_release_doc_lists_verification_steps():
    text = RELEASE_MD.read_text(encoding="utf-8")
    for needle in (
        "pip install agent-policy-gateway",
        "apg-replay --help",
        "apg-bench --help",
    ):
        assert needle in text, (
            f"verification section is missing the {needle!r} check"
        )


# --------------------------------------------------------------------------- #
# Cross-links                                                                 #
# --------------------------------------------------------------------------- #


def test_release_doc_is_in_mkdocs_nav():
    with MKDOCS_YML.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    nav = cfg.get("nav")
    assert isinstance(nav, list), "mkdocs.yml ``nav`` must be a list"

    def walk(items):
        for item in items:
            if isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str):
                        yield value
                    elif isinstance(value, list):
                        yield from walk(value)

    targets = list(walk(nav))
    assert "release.md" in targets, "release.md must appear in mkdocs nav"


def test_readme_has_release_section_linking_to_runbook():
    text = README_MD.read_text(encoding="utf-8")
    assert "## Release" in text, "README must have a ## Release section"
    assert "docs/release.md" in text, "README must link to docs/release.md"


def test_index_links_to_release_doc():
    text = INDEX_MD.read_text(encoding="utf-8")
    assert "release.md" in text, "docs/index.md must reference release.md"
    assert "Cut a release" in text, (
        "docs/index.md should add a 'Cut a release' row to the where-to-start table"
    )


def test_design_doc_records_r14a_note():
    text = DESIGN_MD.read_text(encoding="utf-8")
    assert "Release runbook (R14a)" in text, (
        "docs/design.md should record the R14a design note"
    )


def test_release_doc_relative_links_resolve():
    """Every relative markdown link in release.md must target a real file."""
    text = RELEASE_MD.read_text(encoding="utf-8")
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for target in link_re.findall(text):
        # Skip absolute http(s) URLs, anchors, and mailto links.
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        # Strip fragment / query suffixes if any.
        clean = target.split("#", 1)[0].split("?", 1)[0]
        if not clean:
            continue
        resolved = (RELEASE_MD.parent / clean).resolve()
        assert resolved.exists(), (
            f"release.md links to {target!r} which does not resolve to a file"
        )
