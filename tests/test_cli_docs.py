"""R46: keep ``docs/cli.md`` in lockstep with the ``apg`` argparse parser.

The CLI reference is hand-written but machine-checked. These tests walk the
parser tree built by :func:`agent_policy_gateway.cli._build_parser` and assert
that every subcommand path and every long option string appears verbatim in
``docs/cli.md`` -- and, in the other direction, that every ``--flag`` the page
mentions is a real option of some parser. A flag added, renamed or removed in
``cli.py`` without a docs update fails the suite.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from agent_policy_gateway.cli import _build_parser

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_DOC = REPO_ROOT / "docs" / "cli.md"

#: Long options that legitimately appear in the page but are not options of the
#: ``apg`` parser: argparse's own ``--help``, and the sibling ``apg-replay``
#: flags documented in the closing section.
_ALLOWED_EXTRA_FLAGS = frozenset({"--help", "--limit", "--verify"})

#: Every exit code ``apg`` (or its sibling ``apg-replay``) can return. The page
#: must document each one.
_EXIT_CODES = (0, 1, 2, 3, 4, 5, 6)


# --------------------------------------------------------------------------- #
# Parser introspection                                                        #
# --------------------------------------------------------------------------- #


def _walk(
    parser: argparse.ArgumentParser, prefix: str
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return ``(command_paths, (command_path, long_option) pairs)``.

    ``prefix`` is the invocation path of ``parser`` (e.g. ``"apg policy"``).
    Recurses through every ``add_subparsers`` level.
    """
    commands: list[str] = []
    options: list[tuple[str, str]] = []

    for action in parser._actions:  # noqa: SLF001 - argparse has no public API
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for name, subparser in action.choices.items():
                path = f"{prefix} {name}"
                commands.append(path)
                sub_commands, sub_options = _walk(subparser, path)
                commands.extend(sub_commands)
                options.extend(sub_options)
            continue
        for opt in action.option_strings:
            if opt.startswith("--") and opt != "--help":
                options.append((prefix, opt))

    return commands, options


def _parser_surface() -> tuple[list[str], list[tuple[str, str]]]:
    return _walk(_build_parser(), "apg")


def _leaf_commands(commands: list[str]) -> list[str]:
    """Command paths that are not a prefix of another command path."""
    return [c for c in commands if not any(o.startswith(c + " ") for o in commands)]


def _doc_text() -> str:
    assert CLI_DOC.is_file(), f"missing {CLI_DOC}"
    return CLI_DOC.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The page exists and is wired into the site                                  #
# --------------------------------------------------------------------------- #


class TestPageWiring:
    def test_page_exists_and_is_non_trivial(self) -> None:
        text = _doc_text()
        assert len(text.splitlines()) > 50, "docs/cli.md looks like a stub"

    def test_page_is_in_mkdocs_nav(self) -> None:
        import yaml

        config = yaml.safe_load(
            (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        )
        flat = repr(config["nav"])
        assert "cli.md" in flat, "docs/cli.md must be linked from the mkdocs nav"


# --------------------------------------------------------------------------- #
# Parser -> docs (nothing undocumented)                                       #
# --------------------------------------------------------------------------- #


class TestParserIsDocumented:
    def test_parser_surface_is_non_empty(self) -> None:
        commands, options = _parser_surface()
        assert commands, "parser exposes no subcommands -- introspection broke"
        assert options, "parser exposes no long options -- introspection broke"

    @pytest.mark.parametrize("command", sorted(set(_parser_surface()[0])))
    def test_every_subcommand_documented(self, command: str) -> None:
        assert command in _doc_text(), (
            f"subcommand {command!r} is missing from docs/cli.md"
        )

    @pytest.mark.parametrize(
        ("command", "option"), sorted(set(_parser_surface()[1]))
    )
    def test_every_long_option_documented(self, command: str, option: str) -> None:
        assert option in _doc_text(), (
            f"option {option!r} of {command!r} is missing from docs/cli.md"
        )

    def test_every_leaf_command_has_a_heading(self) -> None:
        text = _doc_text()
        headings = {
            line.lstrip("#").strip().strip("`")
            for line in text.splitlines()
            if line.startswith("#")
        }
        commands, _ = _parser_surface()
        missing = [c for c in _leaf_commands(commands) if c not in headings]
        assert not missing, f"no section heading in docs/cli.md for: {missing}"


# --------------------------------------------------------------------------- #
# Docs -> parser (nothing stale)                                              #
# --------------------------------------------------------------------------- #


class TestDocsMatchParser:
    def test_no_unknown_flags_documented(self) -> None:
        _, options = _parser_surface()
        known = {opt for _cmd, opt in options} | _ALLOWED_EXTRA_FLAGS
        # Only look at inline-code spans, so prose hyphens are never mistaken
        # for flags.
        mentioned = {
            match
            for span in re.findall(r"`([^`]+)`", _doc_text())
            for match in re.findall(r"--[A-Za-z][A-Za-z0-9-]*", span)
        }
        unknown = sorted(mentioned - known)
        assert not unknown, (
            f"docs/cli.md documents flags that no parser defines: {unknown}"
        )


# --------------------------------------------------------------------------- #
# Exit-code table                                                             #
# --------------------------------------------------------------------------- #


class TestExitCodeTable:
    def test_has_an_exit_code_section(self) -> None:
        assert "Exit codes" in _doc_text()

    @pytest.mark.parametrize("code", _EXIT_CODES)
    def test_every_exit_code_documented(self, code: int) -> None:
        table = _doc_text().split("Exit codes", 1)[1]
        assert f"`{code}`" in table, (
            f"exit code {code} is missing from the docs/cli.md exit-code table"
        )
