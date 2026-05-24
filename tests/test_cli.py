"""Tests for the ``apg`` console entry point (R18).

These assert CLI contracts only — exit codes, stdout/stderr content, and the
first-match trace — never timing or environment-specific behaviour. The CLI is
driven through ``main(argv)`` so no subprocess is needed.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from agent_policy_gateway import cli_main
from agent_policy_gateway.cli import _parse_taint, main
from agent_policy_gateway.core import TaintLabel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY = REPO_ROOT / "policies" / "default.yaml"


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run ``main(argv)`` capturing (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestExport:
    def test_cli_main_is_main(self) -> None:
        assert cli_main is main


class TestParseTaint:
    def test_none_is_empty_label(self) -> None:
        assert _parse_taint(None) == TaintLabel()

    def test_empty_string_is_empty_label(self) -> None:
        assert _parse_taint("") == TaintLabel()

    def test_comma_separated_sources(self) -> None:
        assert _parse_taint("web,pii") == TaintLabel.of("web", "pii")

    def test_whitespace_and_blank_fields_ignored(self) -> None:
        assert _parse_taint(" web , , pii ") == TaintLabel.of("web", "pii")


class TestValidate:
    def test_default_policy_is_valid_exit_0(self) -> None:
        rc, out, err = _run(["policy", "validate", str(DEFAULT_POLICY)])
        assert rc == 0
        assert "valid" in out
        assert "default" in out

    def test_missing_file_exits_2(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.yaml"
        rc, out, err = _run(["policy", "validate", str(missing)])
        assert rc == 2
        assert "not found" in err

    def test_malformed_yaml_exits_nonzero_with_line(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        # Broken block mapping: a value that opens a flow seq but never closes.
        bad.write_text("version: 1\nname: x\nrules: [\n", encoding="utf-8")
        rc, out, err = _run(["policy", "validate", str(bad)])
        assert rc != 0
        assert rc == 1
        # The error must be line-located (yaml reports a line/column mark).
        assert "line" in err.lower()

    def test_schema_violation_exits_1(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad2.yaml"
        # Unsupported version trips the Pydantic validator.
        bad.write_text("version: 2\nname: x\nrules: []\n", encoding="utf-8")
        rc, out, err = _run(["policy", "validate", str(bad)])
        assert rc == 1
        assert "invalid policy" in err


class TestExplain:
    def test_web_to_email_matches_deny_rule(self) -> None:
        rc, out, err = _run(
            [
                "policy",
                "explain",
                str(DEFAULT_POLICY),
                "--tool",
                "send_email",
                "--identity",
                "agent.research",
                "--taint",
                "web",
            ]
        )
        assert rc == 0
        assert "deny-web-to-email" in out
        assert "matched rule: deny-web-to-email" in out
        assert "[MATCH ]" in out

    def test_no_match_reports_default(self) -> None:
        rc, out, err = _run(
            [
                "policy",
                "explain",
                str(DEFAULT_POLICY),
                "--tool",
                "unheard_of_tool",
            ]
        )
        assert rc == 0
        assert "no rule matched" in out

    def test_trace_lists_every_rule(self) -> None:
        rc, out, err = _run(
            [
                "policy",
                "explain",
                str(DEFAULT_POLICY),
                "--tool",
                "kb_lookup",
            ]
        )
        assert rc == 0
        # kb_lookup is the third rule; the two earlier ones must show as no-match.
        assert "deny-web-to-email" in out
        assert "review-pii-egress" in out
        assert "matched rule: allow-internal-readers" in out

    def test_explain_missing_file_exits_2(self, tmp_path: Path) -> None:
        rc, out, err = _run(
            ["policy", "explain", str(tmp_path / "nope.yaml"), "--tool", "x"]
        )
        assert rc == 2
        assert "not found" in err

    def test_explain_requires_tool(self) -> None:
        # argparse exits with SystemExit(2) when a required option is missing.
        with pytest.raises(SystemExit):
            _run(["policy", "explain", str(DEFAULT_POLICY)])


class TestDispatch:
    def test_no_subcommand_errors(self) -> None:
        with pytest.raises(SystemExit):
            _run([])

    def test_unknown_group_errors(self) -> None:
        with pytest.raises(SystemExit):
            _run(["frobnicate"])
