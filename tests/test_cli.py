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
from agent_policy_gateway.cli import (
    _coerce_scalar,
    _concretize_glob,
    _parse_taint,
    _pattern_at_least_as_general,
    _taint_at_least_as_general,
    main,
)
from agent_policy_gateway.core import TaintLabel
from agent_policy_gateway.policy import TaintCondition

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


ARG_POLICY = """\
version: 1
name: arg-demo
rules:
  - id: deny-public-channel
    when:
      tool: post_message
      arg_equals: {channel: "#public"}
    effect: {action: deny, reason: "no posting to public channels"}
  - id: allow-rest
    effect: {action: allow}
"""


class TestExplainArgEquals:
    """R25: the explain trace honours ``arg_equals`` and ``--arg``."""

    @pytest.fixture()
    def policy_file(self, tmp_path: Path) -> Path:
        f = tmp_path / "arg-demo.yaml"
        f.write_text(ARG_POLICY, encoding="utf-8")
        return f

    def test_arg_match_hits_rule(self, policy_file: Path) -> None:
        rc, out, err = _run(
            [
                "policy",
                "explain",
                str(policy_file),
                "--tool",
                "post_message",
                "--arg",
                "channel=#public",
            ]
        )
        assert rc == 0
        assert "matched rule: deny-public-channel" in out
        assert "args={'channel': '#public'}" in out

    def test_arg_mismatch_falls_through(self, policy_file: Path) -> None:
        rc, out, err = _run(
            [
                "policy",
                "explain",
                str(policy_file),
                "--tool",
                "post_message",
                "--arg",
                "channel=#random",
            ]
        )
        assert rc == 0
        assert "argument channel='#random' != required '#public'" in out
        assert "matched rule: allow-rest" in out

    def test_missing_arg_is_spelled_out(self, policy_file: Path) -> None:
        rc, out, err = _run(
            ["policy", "explain", str(policy_file), "--tool", "post_message"]
        )
        assert rc == 0
        assert "argument 'channel' is missing (rule needs channel='#public')" in out
        assert "matched rule: allow-rest" in out

    def test_no_args_keeps_legacy_call_line(self, policy_file: Path) -> None:
        rc, out, err = _run(
            ["policy", "explain", str(policy_file), "--tool", "post_message"]
        )
        assert rc == 0
        assert "args=" not in out  # call line keeps its R18 shape without --arg

    def test_repeatable_and_typed_args(self, tmp_path: Path) -> None:
        f = tmp_path / "typed.yaml"
        f.write_text(
            """\
version: 1
name: typed
rules:
  - id: r1
    when:
      arg_equals: {count: 3, dry_run: true}
    effect: {action: deny}
""",
            encoding="utf-8",
        )
        rc, out, err = _run(
            [
                "policy",
                "explain",
                str(f),
                "--tool",
                "anything",
                "--arg",
                "count=3",
                "--arg",
                "dry_run=true",
            ]
        )
        assert rc == 0
        assert "matched rule: r1" in out

    def test_malformed_arg_exits_via_argparse(self, policy_file: Path) -> None:
        with pytest.raises(SystemExit):
            _run(
                [
                    "policy",
                    "explain",
                    str(policy_file),
                    "--tool",
                    "x",
                    "--arg",
                    "no-equals-sign",
                ]
            )

    def test_coerce_scalar(self) -> None:
        assert _coerce_scalar("true") is True
        assert _coerce_scalar("FALSE") is False
        assert _coerce_scalar("3") == 3 and isinstance(_coerce_scalar("3"), int)
        assert _coerce_scalar("#public") == "#public"
        assert _coerce_scalar("") == ""
        assert _coerce_scalar("1.5") == "1.5"  # only decimal ints are coerced


class TestDispatch:
    def test_no_subcommand_errors(self) -> None:
        with pytest.raises(SystemExit):
            _run([])

    def test_unknown_group_errors(self) -> None:
        with pytest.raises(SystemExit):
            _run(["frobnicate"])


# --- ``apg policy diff`` (R24) -------------------------------------------------

OLD_POLICY_YAML = """\
version: 1
name: old-pol
rules:
  - id: deny-web-to-email
    when:
      tool: send_email
      taint: { any_of: [web] }
    effect: { action: deny, reason: "no exfiltration" }
  - id: gate-publish
    when:
      tool: "publish_*"
      identity: agent.research
    effect: { action: review }
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestConcretizeGlob:
    def test_none_returns_default(self) -> None:
        assert _concretize_glob(None, default="d") == "d"
        assert _concretize_glob(None, default=None) is None

    def test_literal_passes_through(self) -> None:
        assert _concretize_glob("send_email", default=None) == "send_email"

    def test_star_glob_is_concretized_and_matches(self) -> None:
        import fnmatch

        sample = _concretize_glob("send_*", default=None)
        assert sample is not None
        assert fnmatch.fnmatchcase(sample, "send_*")

    def test_question_mark_glob(self) -> None:
        import fnmatch

        sample = _concretize_glob("tool_?", default=None)
        assert sample is not None
        assert fnmatch.fnmatchcase(sample, "tool_?")


class TestDiff:
    def test_identical_policies_no_decision_changes(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        new = _write(tmp_path, "new.yaml", OLD_POLICY_YAML)
        rc, out, err = _run(["policy", "diff", str(old), str(new)])
        assert rc == 0
        assert "no decision changes" in out

    def test_same_file_twice_no_decision_changes(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        rc, out, err = _run(["policy", "diff", str(old), str(old)])
        assert rc == 0
        assert "no decision changes" in out

    def test_effect_flip_is_reported(self, tmp_path: Path) -> None:
        flipped = OLD_POLICY_YAML.replace(
            'effect: { action: deny, reason: "no exfiltration" }',
            "effect: { action: allow }",
        )
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        new = _write(tmp_path, "new.yaml", flipped)
        rc, out, err = _run(["policy", "diff", str(old), str(new)])
        assert rc == 0
        assert "deny-web-to-email" in out
        assert "deny" in out
        assert "allow" in out
        assert "old:" in out and "new:" in out

    def test_added_rule_is_reported_as_no_match_to_match(self, tmp_path: Path) -> None:
        added = OLD_POLICY_YAML + (
            "  - id: deny-shell\n"
            "    when:\n"
            "      tool: run_shell\n"
            "    effect: { action: deny }\n"
        )
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        new = _write(tmp_path, "new.yaml", added)
        rc, out, err = _run(["policy", "diff", str(old), str(new)])
        assert rc == 0
        assert "run_shell" in out
        assert "no match" in out
        assert "deny-shell -> deny" in out

    def test_removed_rule_is_reported_as_match_to_no_match(self, tmp_path: Path) -> None:
        added = OLD_POLICY_YAML + (
            "  - id: deny-shell\n"
            "    when:\n"
            "      tool: run_shell\n"
            "    effect: { action: deny }\n"
        )
        old = _write(tmp_path, "old.yaml", added)
        new = _write(tmp_path, "new.yaml", OLD_POLICY_YAML)
        rc, out, err = _run(["policy", "diff", str(old), str(new)])
        assert rc == 0
        assert "deny-shell -> deny" in out
        assert "no match" in out

    def test_header_names_both_policies(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        new = _write(
            tmp_path, "new.yaml", OLD_POLICY_YAML.replace("name: old-pol", "name: new-pol")
        )
        rc, out, err = _run(["policy", "diff", str(old), str(new)])
        assert rc == 0
        assert "old-pol" in out
        assert "new-pol" in out

    def test_single_scenario_via_tool_flag(self, tmp_path: Path) -> None:
        flipped = OLD_POLICY_YAML.replace(
            'effect: { action: deny, reason: "no exfiltration" }',
            "effect: { action: allow }",
        )
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        new = _write(tmp_path, "new.yaml", flipped)
        rc, out, err = _run(
            [
                "policy",
                "diff",
                str(old),
                str(new),
                "--tool",
                "send_email",
                "--taint",
                "web",
            ]
        )
        assert rc == 0
        assert "1 scenario(s)" in out
        assert "deny-web-to-email" in out

    def test_single_scenario_can_report_no_changes(self, tmp_path: Path) -> None:
        flipped = OLD_POLICY_YAML.replace(
            'effect: { action: deny, reason: "no exfiltration" }',
            "effect: { action: allow }",
        )
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        new = _write(tmp_path, "new.yaml", flipped)
        # An untainted call never hit the flipped rule in either policy.
        rc, out, err = _run(
            ["policy", "diff", str(old), str(new), "--tool", "kb_lookup"]
        )
        assert rc == 0
        assert "no decision changes" in out

    def test_identity_scenarios_come_from_selectors(self, tmp_path: Path) -> None:
        # gate-publish flips review -> deny; the matrix scenario for it carries
        # the selector's identity, so the change is visible without flags.
        flipped = OLD_POLICY_YAML.replace(
            "effect: { action: review }", "effect: { action: deny }"
        )
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        new = _write(tmp_path, "new.yaml", flipped)
        rc, out, err = _run(["policy", "diff", str(old), str(new)])
        assert rc == 0
        assert "agent.research" in out
        assert "gate-publish -> review" in out
        assert "gate-publish -> deny" in out

    def test_missing_old_file_exits_2(self, tmp_path: Path) -> None:
        new = _write(tmp_path, "new.yaml", OLD_POLICY_YAML)
        rc, out, err = _run(["policy", "diff", str(tmp_path / "nope.yaml"), str(new)])
        assert rc == 2
        assert "old policy file not found" in err

    def test_missing_new_file_exits_2(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        rc, out, err = _run(["policy", "diff", str(old), str(tmp_path / "nope.yaml")])
        assert rc == 2
        assert "new policy file not found" in err

    def test_malformed_new_policy_exits_1(self, tmp_path: Path) -> None:
        old = _write(tmp_path, "old.yaml", OLD_POLICY_YAML)
        bad = _write(tmp_path, "bad.yaml", "version: 2\nname: x\nrules: []\n")
        rc, out, err = _run(["policy", "diff", str(old), str(bad)])
        assert rc == 1
        assert "invalid new policy" in err

    def test_malformed_old_policy_exits_1(self, tmp_path: Path) -> None:
        new = _write(tmp_path, "new.yaml", OLD_POLICY_YAML)
        bad = _write(tmp_path, "bad.yaml", "rules: [\n")
        rc, out, err = _run(["policy", "diff", str(bad), str(new)])
        assert rc == 1
        assert "invalid old policy" in err

    def test_default_policy_diffed_against_itself(self) -> None:
        rc, out, err = _run(
            ["policy", "diff", str(DEFAULT_POLICY), str(DEFAULT_POLICY)]
        )
        assert rc == 0
        assert "no decision changes" in out


# --- ``apg policy lint`` (R26) -------------------------------------------------

CLEAN_LINT_YAML = """\
version: 1
name: clean-pol
rules:
  - id: deny-web-to-email
    when:
      tool: send_email
      taint: { any_of: [web] }
    effect: { action: deny }
  - id: allow-research-publish
    when:
      tool: "publish_*"
      identity: agent.research
    effect: { action: allow }
  - id: deny-publish-from-other-agents
    when:
      tool: "publish_*"
    effect: { action: deny }
"""


class TestPatternGenerality:
    def test_unset_earlier_is_most_general(self) -> None:
        assert _pattern_at_least_as_general(None, "send_email")
        assert _pattern_at_least_as_general(None, None)

    def test_unset_later_is_not_subsumed_by_set_earlier(self) -> None:
        assert not _pattern_at_least_as_general("send_email", None)

    def test_equal_patterns_subsume(self) -> None:
        assert _pattern_at_least_as_general("send_*", "send_*")

    def test_glob_subsumes_literal(self) -> None:
        assert _pattern_at_least_as_general("send_*", "send_email")
        assert not _pattern_at_least_as_general("send_*", "kb_lookup")

    def test_distinct_globs_conservatively_not_compared(self) -> None:
        # ``send_*`` really does subsume ``send_e*`` but the conservative
        # check refuses to reason about glob-vs-glob pairs.
        assert not _pattern_at_least_as_general("send_*", "send_e*")


class TestTaintGenerality:
    def test_empty_earlier_is_most_general(self) -> None:
        later = TaintCondition(any_of=("web",))
        assert _taint_at_least_as_general(None, later)
        assert _taint_at_least_as_general(TaintCondition(), later)

    def test_constrained_earlier_does_not_subsume_empty_later(self) -> None:
        assert not _taint_at_least_as_general(TaintCondition(any_of=("web",)), None)

    def test_any_of_superset_subsumes(self) -> None:
        earlier = TaintCondition(any_of=("web", "pii"))
        later = TaintCondition(any_of=("web",))
        assert _taint_at_least_as_general(earlier, later)
        assert not _taint_at_least_as_general(later, earlier)

    def test_any_of_guaranteed_by_later_all_of(self) -> None:
        earlier = TaintCondition(any_of=("web",))
        later = TaintCondition(all_of=("web", "pii"))
        assert _taint_at_least_as_general(earlier, later)

    def test_none_of_must_be_subset(self) -> None:
        earlier = TaintCondition(none_of=("secret",))
        later = TaintCondition(none_of=("secret", "pii"))
        assert _taint_at_least_as_general(earlier, later)
        assert not _taint_at_least_as_general(later, earlier)


class TestLint:
    def test_clean_policy_exits_0(self, tmp_path: Path) -> None:
        f = _write(tmp_path, "clean.yaml", CLEAN_LINT_YAML)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 0
        assert "no lint findings" in out
        assert "clean-pol" in out

    def test_shipped_policies_are_lint_clean(self) -> None:
        policies_dir = REPO_ROOT / "policies"
        files = sorted(policies_dir.glob("*.yaml"))
        assert files, "no shipped policies found"
        for f in files:
            rc, out, err = _run(["policy", "lint", str(f)])
            assert rc == 0, f"{f.name}: {out}{err}"

    def test_identical_selector_is_shadowed(self, tmp_path: Path) -> None:
        yaml_text = CLEAN_LINT_YAML + (
            "  - id: dead-duplicate\n"
            "    when:\n"
            "      tool: send_email\n"
            "      taint: { any_of: [web] }\n"
            "    effect: { action: allow }\n"
        )
        f = _write(tmp_path, "dup.yaml", yaml_text)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 3
        assert "W001" in out
        assert "'dead-duplicate' is shadowed by earlier rule 'deny-web-to-email'" in out
        assert "1 lint finding(s)" in out

    def test_catch_all_shadows_everything_after_it(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\n"
            "name: catch-all-pol\n"
            "rules:\n"
            "  - id: allow-everything\n"
            "    effect: { action: allow }\n"
            "  - id: never-reached\n"
            "    when: { tool: send_email }\n"
            "    effect: { action: deny }\n"
        )
        f = _write(tmp_path, "catchall.yaml", yaml_text)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 3
        assert "'never-reached' is shadowed by earlier rule 'allow-everything'" in out

    def test_glob_shadows_literal_tool(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\n"
            "name: glob-pol\n"
            "rules:\n"
            "  - id: deny-all-sends\n"
            "    when: { tool: \"send_*\" }\n"
            "    effect: { action: deny }\n"
            "  - id: allow-send-email\n"
            "    when: { tool: send_email }\n"
            "    effect: { action: allow }\n"
        )
        f = _write(tmp_path, "glob.yaml", yaml_text)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 3
        assert "'allow-send-email' is shadowed by earlier rule 'deny-all-sends'" in out

    def test_narrower_identity_earlier_does_not_shadow(self, tmp_path: Path) -> None:
        # CLEAN_LINT_YAML's allow-research-publish (identity-constrained) must
        # not be reported as shadowing deny-publish-from-other-agents.
        f = _write(tmp_path, "clean.yaml", CLEAN_LINT_YAML)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 0

    def test_extra_arg_equals_later_is_shadowed(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\n"
            "name: args-pol\n"
            "rules:\n"
            "  - id: deny-public-posts\n"
            "    when: { tool: post_message }\n"
            "    effect: { action: deny }\n"
            "  - id: review-public-channel\n"
            "    when:\n"
            "      tool: post_message\n"
            "      arg_equals: { channel: \"#public\" }\n"
            "    effect: { action: review }\n"
        )
        f = _write(tmp_path, "args.yaml", yaml_text)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 3
        assert "'review-public-channel' is shadowed" in out

    def test_all_of_none_of_contradiction_reported(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\n"
            "name: contra-pol\n"
            "rules:\n"
            "  - id: impossible\n"
            "    when:\n"
            "      tool: send_email\n"
            "      taint: { all_of: [web], none_of: [web] }\n"
            "    effect: { action: deny }\n"
        )
        f = _write(tmp_path, "contra.yaml", yaml_text)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 3
        assert "W002" in out
        assert "'impossible' can never match" in out
        assert "all_of" in out and "none_of" in out

    def test_any_of_swallowed_by_none_of_reported(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\n"
            "name: contra2-pol\n"
            "rules:\n"
            "  - id: impossible-any\n"
            "    when:\n"
            "      taint: { any_of: [web, pii], none_of: [web, pii, secret] }\n"
            "    effect: { action: deny }\n"
        )
        f = _write(tmp_path, "contra2.yaml", yaml_text)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 3
        assert "W002" in out
        assert "'impossible-any' can never match" in out

    def test_contradictory_rule_does_not_shadow(self, tmp_path: Path) -> None:
        # The impossible catch-all-ish rule never matches anything, so the
        # rule after it must NOT be reported as shadowed by it.
        yaml_text = (
            "version: 1\n"
            "name: contra3-pol\n"
            "rules:\n"
            "  - id: impossible\n"
            "    when:\n"
            "      taint: { all_of: [web], none_of: [web] }\n"
            "    effect: { action: deny }\n"
            "  - id: live-rule\n"
            "    when: { tool: send_email }\n"
            "    effect: { action: allow }\n"
        )
        f = _write(tmp_path, "contra3.yaml", yaml_text)
        rc, out, err = _run(["policy", "lint", str(f)])
        assert rc == 3
        assert "W002" in out
        assert "shadowed" not in out

    def test_missing_file_exits_2(self, tmp_path: Path) -> None:
        rc, out, err = _run(["policy", "lint", str(tmp_path / "nope.yaml")])
        assert rc == 2
        assert "not found" in err

    def test_malformed_policy_exits_1(self, tmp_path: Path) -> None:
        bad = _write(tmp_path, "bad.yaml", "version: 2\nname: x\nrules: []\n")
        rc, out, err = _run(["policy", "lint", str(bad)])
        assert rc == 1
        assert "invalid policy" in err
