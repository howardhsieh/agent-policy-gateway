"""Tests for the R13 project threat-model document.

These tests lock in the structural invariants of ``docs/threat-model.md``:
the file exists, is non-trivial in size, uses ATX headings, contains
the canonical top-level sections, is wired into the mkdocs ``nav``,
is referenced from ``README.md`` and ``docs/index.md``, and every
relative markdown link target on the page resolves to a file that
actually lives in the repo.

The tests deliberately do *not* assert prose. The threat model is a
living document and editorial freedom matters; only structure is
enforced here. ``tests/test_docs_site.py`` separately covers the
mkdocs build, so this file does not duplicate that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
THREAT_MODEL = DOCS_DIR / "threat-model.md"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
README = REPO_ROOT / "README.md"
INDEX_MD = DOCS_DIR / "index.md"


REQUIRED_SECTIONS = (
    "## Assets",
    "## Trust boundaries",
    "## Adversary classes",
    "## Assumptions",
    "## In-scope attacker capabilities",
    "## Abuse scenarios and mitigations",
    "## Residual risks",
    "## Out of scope",
)

# Adversary IDs the document promises to enumerate. The threat model
# is allowed to grow more, but these eight are the project's current
# operating set and removing any of them is a breaking change worth a
# roadmap item.
REQUIRED_ADVERSARY_IDS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8")


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def threat_model_text() -> str:
    assert THREAT_MODEL.is_file(), f"missing {THREAT_MODEL}"
    return THREAT_MODEL.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# File presence and shape                                                     #
# --------------------------------------------------------------------------- #


def test_threat_model_file_exists() -> None:
    assert THREAT_MODEL.is_file(), f"missing {THREAT_MODEL}"


def test_threat_model_is_substantial(threat_model_text: str) -> None:
    # The threat model is a project-level commitment; a stub would be
    # worse than nothing. 1500+ characters is a lower bound that
    # rejects a placeholder without constraining future editing.
    assert len(threat_model_text) > 1500


def test_threat_model_starts_with_atx_h1(threat_model_text: str) -> None:
    first_line = threat_model_text.splitlines()[0]
    assert first_line.startswith("# "), (
        "threat-model.md must start with an ATX H1; "
        f"first line was {first_line!r}"
    )


def test_threat_model_uses_atx_headings_only(threat_model_text: str) -> None:
    # No setext headings (=== / --- under a line). Mkdocs renders both,
    # but the project standard is ATX so the test_docs_site grep-style
    # checks elsewhere stay simple.
    lines = threat_model_text.splitlines()
    for prev, curr in zip(lines, lines[1:], strict=False):
        if prev.strip() and curr and set(curr.strip()) in ({"="}, {"-"}):
            # A setext underline must be *only* = or -.
            # Allow `---` separators that appear after a blank line.
            if prev.strip() and not prev.startswith(("#", "-", "|", "*", ">")):
                pytest.fail(
                    f"setext heading detected near line: {prev!r}\n{curr!r}"
                )


# --------------------------------------------------------------------------- #
# Required sections                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_threat_model_contains_required_section(
    threat_model_text: str, section: str
) -> None:
    # Match the heading on its own line so a substring in prose does
    # not falsely satisfy the check.
    pattern = rf"(?m)^{re.escape(section)}\s*$"
    assert re.search(pattern, threat_model_text), (
        f"required section heading missing: {section!r}"
    )


def test_required_sections_appear_in_documented_order(
    threat_model_text: str,
) -> None:
    indices = []
    for section in REQUIRED_SECTIONS:
        pattern = rf"(?m)^{re.escape(section)}\s*$"
        match = re.search(pattern, threat_model_text)
        assert match, f"required section missing: {section!r}"
        indices.append(match.start())
    assert indices == sorted(indices), (
        "required sections must appear in the documented order; got "
        f"{indices!r}"
    )


@pytest.mark.parametrize("adv_id", REQUIRED_ADVERSARY_IDS)
def test_threat_model_enumerates_adversary_id(
    threat_model_text: str, adv_id: str
) -> None:
    # Adversary lines look like "**A1 — Indirect prompt injection..."
    # but the test only requires the ID token "A1" appears somewhere
    # in the file — formatting is editorial.
    assert adv_id in threat_model_text, (
        f"adversary id {adv_id!r} not enumerated in threat-model.md"
    )


# --------------------------------------------------------------------------- #
# Wiring: mkdocs nav, README, docs index                                      #
# --------------------------------------------------------------------------- #


def test_threat_model_listed_in_mkdocs_nav() -> None:
    assert MKDOCS_YML.is_file(), f"missing {MKDOCS_YML}"
    cfg = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    nav = cfg.get("nav")
    assert isinstance(nav, list) and nav, "mkdocs.yml has no nav"

    def walk(node: object) -> list[str]:
        out: list[str] = []
        if isinstance(node, list):
            for item in node:
                out.extend(walk(item))
        elif isinstance(node, dict):
            for value in node.values():
                if isinstance(value, str):
                    out.append(value)
                else:
                    out.extend(walk(value))
        elif isinstance(node, str):
            out.append(node)
        return out

    refs = walk(nav)
    assert "threat-model.md" in refs, (
        f"threat-model.md not referenced from mkdocs nav; got {refs!r}"
    )


def test_readme_links_to_threat_model() -> None:
    assert README.is_file()
    text = README.read_text(encoding="utf-8")
    assert "docs/threat-model.md" in text, (
        "README.md must link to docs/threat-model.md"
    )


def test_docs_index_links_to_threat_model() -> None:
    assert INDEX_MD.is_file()
    text = INDEX_MD.read_text(encoding="utf-8")
    assert "threat-model.md" in text, (
        "docs/index.md must reference threat-model.md"
    )


# --------------------------------------------------------------------------- #
# Internal-link integrity                                                     #
# --------------------------------------------------------------------------- #


# Match standard markdown link targets like [text](path/to/file.md#anchor).
# Excludes inline code and reference-style links to keep the test simple
# and predictable.
_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _internal_targets(text: str) -> list[str]:
    targets: list[str] = []
    for match in _LINK_RE.finditer(text):
        target = match.group(1)
        if not target:
            continue
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(target)
    return targets


def test_threat_model_internal_links_resolve(threat_model_text: str) -> None:
    # Resolve every relative markdown link target on the page against
    # the docs/ directory (the page lives in docs/), stripping any
    # ``#anchor`` fragment.
    bad: list[tuple[str, Path]] = []
    for target in _internal_targets(threat_model_text):
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        candidate = (DOCS_DIR / path_part).resolve()
        if not candidate.exists():
            # Also tolerate paths relative to the repo root (some
            # docs link to things like `../examples/...`).
            alt = (REPO_ROOT / path_part).resolve()
            if alt.exists():
                continue
            bad.append((target, candidate))
    assert not bad, (
        "broken internal markdown links in threat-model.md: "
        + ", ".join(f"{t!r} -> {p}" for t, p in bad)
    )


# --------------------------------------------------------------------------- #
# Cross-doc integrity                                                         #
# --------------------------------------------------------------------------- #


def test_design_doc_mentions_threat_model_section() -> None:
    design = (DOCS_DIR / "design.md").read_text(encoding="utf-8")
    assert "Threat model (R13)" in design, (
        "design.md should record the R13 threat-model design note"
    )
    assert "threat-model.md" in design
