"""Tests for the Progent symbolic-rule import PoC (R54).

Three layers:

* schema/parsing contracts for the JSON form of Progent's runtime
  mapping ``{tool: [[priority, effect, condition, fallback], ...]}``;
* translation contracts — ordering, effect/fallback mapping, enum
  expansion, the hard-allow and leaky-fallback quirks, defaults, and
  the ``policy_to_yaml`` round trip;
* a decision-fidelity harness: a reference evaluator ported from
  upstream ``secagent/tool.py`` ``_check_tool_call`` is run against the
  converted APG policy over a battery of calls, and the two must agree
  wherever every constrained argument is present (the documented
  supported envelope — the absent-argument divergence has its own
  explicit test).
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from agent_policy_gateway import (
    Action,
    ProgentImportError,
    ProgentRule,
    ToolCall,
    Verdict,
    convert_progent_policy,
    load_policy_str,
    load_progent_policy,
    load_progent_policy_str,
    parse_progent_policy,
    policy_to_yaml,
    progent_sorted,
)
from agent_policy_gateway.cli import main
from agent_policy_gateway.progent_import import (
    FALLBACK_CONFIRM,
    FALLBACK_ERROR,
    FALLBACK_TERMINATE,
    HARD_PRIORITY,
    MAX_ENUM_EXPANSION,
    PROGENT_ALLOW,
    PROGENT_FORBID,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _convert(data: dict, **kwargs):
    return convert_progent_policy(parse_progent_policy(data), **kwargs)


def _action(policy, tool: str, args: dict) -> str | None:
    """First-match action value for a call, or None when nothing matches."""
    rule = policy.first_match(ToolCall(tool_name=tool, args=args))
    return None if rule is None else rule.effect.action.value


# ----- parsing -------------------------------------------------------------


class TestParsing:
    def test_minimal_policy_parses(self) -> None:
        parsed = parse_progent_policy({"t": [[0, 0, {}, 0]]})
        assert parsed == {
            "t": [ProgentRule(priority=0, effect=0, condition={}, fallback=0)]
        }

    def test_top_level_must_be_mapping(self) -> None:
        with pytest.raises(ProgentImportError, match="top-level"):
            parse_progent_policy([["t", []]])

    def test_tool_names_must_be_nonempty_strings(self) -> None:
        with pytest.raises(ProgentImportError, match="tool names"):
            parse_progent_policy({"  ": []})

    def test_rules_must_be_a_list(self) -> None:
        with pytest.raises(ProgentImportError, match="list of rules"):
            parse_progent_policy({"t": {}})

    def test_rule_must_have_four_elements(self) -> None:
        with pytest.raises(ProgentImportError, match="4 elements"):
            parse_progent_policy({"t": [[0, 0, {}]]})

    def test_self_updating_rules_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="need_update_policies"):
            parse_progent_policy({"t": [[0, 0, {}, 0, {"t": []}]]})

    @pytest.mark.parametrize("priority", [True, "0", 1.5, None])
    def test_priority_must_be_int(self, priority) -> None:
        with pytest.raises(ProgentImportError, match="priority"):
            parse_progent_policy({"t": [[priority, 0, {}, 0]]})

    @pytest.mark.parametrize("effect", [2, -1, True, "allow"])
    def test_effect_must_be_0_or_1(self, effect) -> None:
        with pytest.raises(ProgentImportError, match="effect"):
            parse_progent_policy({"t": [[0, effect, {}, 0]]})

    @pytest.mark.parametrize("fallback", [3, -1, True, "deny"])
    def test_fallback_must_be_0_1_or_2(self, fallback) -> None:
        with pytest.raises(ProgentImportError, match="fallback"):
            parse_progent_policy({"t": [[0, 0, {}, fallback]]})

    def test_condition_must_be_dict(self) -> None:
        with pytest.raises(ProgentImportError, match="condition"):
            parse_progent_policy({"t": [[0, 0, ["recipient"], 0]]})

    def test_restriction_must_be_str_or_dict(self) -> None:
        with pytest.raises(ProgentImportError, match="callables"):
            parse_progent_policy({"t": [[0, 0, {"a": 3}, 0]]})

    def test_load_str_reports_source_on_bad_json(self) -> None:
        with pytest.raises(ProgentImportError, match="my.json: invalid JSON"):
            load_progent_policy_str("{", source="my.json")

    def test_load_str_reports_source_on_bad_shape(self) -> None:
        with pytest.raises(ProgentImportError, match="my.json: top-level"):
            load_progent_policy_str("[]", source="my.json")

    def test_load_from_file(self, tmp_path: Path) -> None:
        p = tmp_path / "rules.json"
        p.write_text(json.dumps({"t": [[0, 0, {}, 0]]}), encoding="utf-8")
        assert "t" in load_progent_policy(p)


# ----- ordering ------------------------------------------------------------


class TestOrdering:
    def test_lower_priority_first(self) -> None:
        rules = [
            ProgentRule(priority=2, effect=0, condition={}, fallback=0),
            ProgentRule(priority=0, effect=0, condition={}, fallback=0),
        ]
        assert [r.priority for r in progent_sorted(rules)] == [0, 2]

    def test_forbid_before_allow_at_equal_priority(self) -> None:
        rules = [
            ProgentRule(priority=1, effect=PROGENT_ALLOW, condition={}, fallback=0),
            ProgentRule(priority=1, effect=PROGENT_FORBID, condition={}, fallback=0),
        ]
        assert [r.effect for r in progent_sorted(rules)] == [
            PROGENT_FORBID,
            PROGENT_ALLOW,
        ]

    def test_converted_rule_order_follows_progent_sort(self) -> None:
        policy = _convert(
            {
                "t": [
                    [1, 0, {"a": {"const": "x"}}, 0],
                    [0, 1, {"a": {"const": "y"}}, 0],
                ]
            }
        )
        ids = [r.id for r in policy.rules]
        # The forbid (priority 0) precedes the allow (priority 1).
        assert ids.index("progent:t:0") < ids.index("progent:t:1")
        assert policy.rules[ids.index("progent:t:0")].effect.action == Action.DENY


# ----- restriction translation ---------------------------------------------


class TestRestrictions:
    def test_bare_string_is_match_anchored(self) -> None:
        policy = _convert({"t": [[0, 1, {"a": "US.*"}, 0], [1, 0, {}, 0]]})
        assert _action(policy, "t", {"a": "US99"}) == "deny"
        # re.match anchors at the start: a mid-string hit must not fire.
        assert _action(policy, "t", {"a": "AUS99"}) == "allow"

    def test_pattern_keeps_search_semantics(self) -> None:
        policy = _convert({"t": [[0, 1, {"a": {"pattern": "US"}}, 0], [1, 0, {}, 0]]})
        # JSON Schema pattern is re.search: a mid-string hit fires.
        assert _action(policy, "t", {"a": "AUS99"}) == "deny"

    def test_const_becomes_arg_equals(self) -> None:
        policy = _convert({"t": [[0, 0, {"a": {"const": 5}}, 0]]})
        assert policy.rules[0].when.arg_equals == {"a": 5}
        assert _action(policy, "t", {"a": 5}) == "allow"
        assert _action(policy, "t", {"a": 6}) == "deny"

    def test_enum_expands_to_one_rule_per_value(self) -> None:
        policy = _convert({"t": [[0, 0, {"a": {"enum": ["x", "y"]}}, 0]]})
        ids = [r.id for r in policy.rules]
        assert "progent:t:0:0" in ids and "progent:t:0:1" in ids
        assert _action(policy, "t", {"a": "x"}) == "allow"
        assert _action(policy, "t", {"a": "y"}) == "allow"
        assert _action(policy, "t", {"a": "z"}) == "deny"

    def test_enum_cross_product_expansion(self) -> None:
        policy = _convert(
            {"t": [[0, 0, {"a": {"enum": [1, 2]}, "b": {"enum": [3, 4]}}, 0]]}
        )
        combos = {
            (r.when.arg_equals["a"], r.when.arg_equals["b"])
            for r in policy.rules
            if r.when.arg_equals
        }
        assert combos == {(1, 3), (1, 4), (2, 3), (2, 4)}

    def test_expansion_over_cap_fails_loudly(self) -> None:
        values = list(range(MAX_ENUM_EXPANSION + 1))
        with pytest.raises(ProgentImportError, match="cap"):
            _convert({"t": [[0, 0, {"a": {"enum": values}}, 0]]})

    def test_const_intersects_enum(self) -> None:
        policy = _convert(
            {"t": [[0, 0, {"a": {"enum": ["x", "y"], "const": "x"}}, 0]]}
        )
        equals = [r.when.arg_equals for r in policy.rules if r.when.arg_equals]
        assert equals == [{"a": "x"}]

    def test_disjoint_const_enum_never_fires(self) -> None:
        policy = _convert(
            {
                "t": [
                    [0, 0, {"a": {"enum": ["x"], "const": "z"}}, 0],
                    [1, 0, {"a": {"const": "q"}}, 0],
                ]
            }
        )
        # The unsatisfiable allow emits no rule; the later allow still works.
        assert _action(policy, "t", {"a": "q"}) == "allow"
        assert _action(policy, "t", {"a": "x"}) == "deny"

    def test_pattern_filters_enum_values(self) -> None:
        policy = _convert(
            {"t": [[0, 0, {"a": {"enum": ["ax", "bx", 7], "pattern": "^a"}}, 0]]}
        )
        equals = [r.when.arg_equals["a"] for r in policy.rules if r.when.arg_equals]
        # JSON Schema string keywords do not apply to non-strings: 7 survives.
        assert sorted(equals, key=str) == [7, "ax"]

    def test_type_string_alone_means_any_string(self) -> None:
        policy = _convert({"t": [[0, 0, {"a": {"type": "string"}}, 0]]})
        assert policy.rules[0].when.arg_matches == {"a": ""}
        assert _action(policy, "t", {"a": "anything"}) == "allow"
        assert _action(policy, "t", {"a": 5}) == "deny"

    def test_empty_schema_is_vacuous(self) -> None:
        policy = _convert({"t": [[0, 0, {"a": {}}, 0]]})
        sel = policy.rules[0].when
        assert sel.arg_equals is None and sel.arg_matches is None

    def test_non_string_type_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="only 'string'"):
            _convert({"t": [[0, 0, {"a": {"type": "integer"}}, 0]]})

    def test_unsupported_keyword_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="minLength"):
            _convert({"t": [[0, 0, {"a": {"minLength": 3}}, 0]]})

    def test_float_scalar_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="scalar"):
            _convert({"t": [[0, 0, {"a": {"const": 1.5}}, 0]]})

    def test_null_scalar_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="scalar"):
            _convert({"t": [[0, 0, {"a": {"enum": [None]}}, 0]]})

    def test_invalid_regex_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="regex"):
            _convert({"t": [[0, 0, {"a": "("}, 0]]})

    def test_non_string_pattern_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="pattern"):
            _convert({"t": [[0, 0, {"a": {"pattern": 3}}, 0]]})

    def test_empty_enum_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="non-empty"):
            _convert({"t": [[0, 0, {"a": {"enum": []}}, 0]]})

    def test_bool_const_is_type_strict(self) -> None:
        policy = _convert({"t": [[0, 0, {"a": {"const": True}}, 0]]})
        assert _action(policy, "t", {"a": True}) == "allow"
        assert _action(policy, "t", {"a": 1}) == "deny"


# ----- effect / fallback mapping -------------------------------------------


class TestEffects:
    def test_forbid_fallback_error_is_deny(self) -> None:
        policy = _convert({"t": [[0, 1, {}, FALLBACK_ERROR]]})
        assert policy.rules[0].effect.action == Action.DENY

    def test_forbid_fallback_terminate_is_deny_with_note(self) -> None:
        policy = _convert({"t": [[0, 1, {}, FALLBACK_TERMINATE]]})
        assert policy.rules[0].effect.action == Action.DENY
        assert "terminate" in policy.rules[0].effect.reason

    def test_forbid_fallback_confirm_is_review(self) -> None:
        policy = _convert({"t": [[0, 1, {}, FALLBACK_CONFIRM]]})
        assert policy.rules[0].effect.action == Action.REVIEW

    def test_unconditional_rule_is_terminal(self) -> None:
        policy = _convert({"t": [[0, 0, {}, 0], [1, 1, {}, 0]]})
        # The dead later forbid and the per-tool default are both omitted.
        assert [r.id for r in policy.rules] == ["progent:t:0", "progent:default"]


class TestHardAllow:
    def test_hard_allow_denies_everything_else(self) -> None:
        policy = _convert(
            {
                "t": [
                    [HARD_PRIORITY, 0, {"a": {"const": "x"}}, FALLBACK_ERROR],
                    [101, 0, {"a": {"const": "y"}}, 0],
                ]
            }
        )
        assert _action(policy, "t", {"a": "x"}) == "allow"
        # Progent raises inside the hard rule before the later allow runs.
        assert _action(policy, "t", {"a": "y"}) == "deny"
        matched = policy.first_match(ToolCall(tool_name="t", args={"a": "y"}))
        assert matched is not None and matched.id == "progent:t:0:otherwise"

    def test_non_hard_priority_has_no_otherwise_tail(self) -> None:
        policy = _convert({"t": [[0, 0, {"a": {"const": "x"}}, FALLBACK_ERROR]]})
        assert not any(r.id.endswith(":otherwise") for r in policy.rules)

    def test_hard_priority_with_confirm_fallback_is_not_hard(self) -> None:
        policy = _convert(
            {"t": [[HARD_PRIORITY, 0, {"a": {"const": "x"}}, FALLBACK_CONFIRM]]}
        )
        assert not any(r.id.endswith(":otherwise") for r in policy.rules)

    def test_unconditional_hard_allow_emits_no_dead_tail(self) -> None:
        policy = _convert({"t": [[HARD_PRIORITY, 0, {}, FALLBACK_ERROR]]})
        assert [r.id for r in policy.rules] == ["progent:t:0", "progent:default"]


# ----- defaults ------------------------------------------------------------


class TestDefaults:
    def test_fall_off_the_end_uses_last_rules_fallback(self) -> None:
        # Upstream's leaky loop variable: the last examined rule's fallback
        # decides how a no-match call is handled.
        policy = _convert({"t": [[0, 0, {"a": {"const": "x"}}, FALLBACK_CONFIRM]]})
        assert _action(policy, "t", {"a": "nope"}) == "review"
        policy = _convert({"t": [[0, 0, {"a": {"const": "x"}}, FALLBACK_ERROR]]})
        assert _action(policy, "t", {"a": "nope"}) == "deny"

    def test_empty_rule_list_denies_the_tool(self) -> None:
        policy = _convert({"t": []})
        assert _action(policy, "t", {}) == "deny"

    def test_unknown_tool_denied_by_global_default(self) -> None:
        policy = _convert({"t": [[0, 0, {}, 0]]})
        assert _action(policy, "other", {}) == "deny"
        assert policy.rules[-1].id == "progent:default"

    def test_per_tool_default_omits_global_catch_all(self) -> None:
        policy = _convert({"t": [[0, 0, {"a": {"const": "x"}}, 0]]}, default="per-tool")
        assert all(r.id != "progent:default" for r in policy.rules)
        # Governed tool still falls closed; unrelated tools are untouched.
        assert _action(policy, "t", {"a": "nope"}) == "deny"
        assert _action(policy, "other", {}) is None

    def test_unknown_default_mode_rejected(self) -> None:
        with pytest.raises(ProgentImportError, match="default"):
            _convert({"t": []}, default="allow")

    def test_glob_metacharacters_in_tool_names_are_literal(self) -> None:
        policy = _convert({"send_*": [[0, 0, {}, 0]]}, default="per-tool")
        assert _action(policy, "send_*", {}) == "allow"
        # The entry must not govern other tools matching the glob text.
        assert _action(policy, "send_money", {}) is None


# ----- YAML round trip ------------------------------------------------------


class TestPolicyToYaml:
    @pytest.mark.parametrize(
        "data",
        [
            {"t": [[0, 0, {}, 0]]},
            {"t": [[0, 1, {"a": "US.*"}, 2], [1, 0, {"a": {"enum": [1, 2]}}, 0]]},
            {
                "send_money": [
                    [0, 1, {"recipient": {"pattern": "^US"}}, 1],
                    [HARD_PRIORITY, 0, {"recipient": {"const": "UK1"}}, 0],
                ]
            },
        ],
    )
    def test_round_trip(self, data: dict) -> None:
        policy = _convert(data)
        assert load_policy_str(policy_to_yaml(policy)) == policy

    def test_empty_selector_and_reasons_are_pruned(self) -> None:
        policy = _convert({"t": [[0, 0, {}, 0]]})
        text = policy_to_yaml(policy)
        assert "when: {}" not in text
        assert "null" not in text
        assert "reason: ''" not in text

    def test_example_policies_round_trip_too(self) -> None:
        # policy_to_yaml is generic: the shipped policies survive it.
        from agent_policy_gateway import load_policy

        for path in sorted((REPO_ROOT / "policies").glob("*.yaml")):
            policy = load_policy(path)
            assert load_policy_str(policy_to_yaml(policy)) == policy, path.name


# ----- decision fidelity against a reference evaluator ----------------------


class _RefInvalid(Exception):
    pass


def _ref_check_arg(value: object, restriction) -> None:
    """The supported-subset restriction check, JSON Schema semantics."""
    if isinstance(restriction, str):
        if not isinstance(value, str) or re.match(restriction, value) is None:
            raise _RefInvalid
        return
    if restriction.get("type") == "string" and not isinstance(value, str):
        raise _RefInvalid
    def _eq(a: object, b: object) -> bool:
        return isinstance(a, bool) == isinstance(b, bool) and a == b
    if "const" in restriction and not _eq(value, restriction["const"]):
        raise _RefInvalid
    if "enum" in restriction and not any(
        _eq(value, v) for v in restriction["enum"]
    ):
        raise _RefInvalid
    if (
        "pattern" in restriction
        and isinstance(value, str)
        and re.search(restriction["pattern"], value) is None
    ):
        raise _RefInvalid


def _reference_decide(policy_map: dict, tool: str, kwargs: dict) -> str:
    """Port of upstream ``_check_tool_call`` for the supported subset.

    Returns ``allow`` / ``deny`` / ``review`` (``review`` standing in for
    both interactive-confirm paths, firing forbid and fall-off-the-end).
    """
    rules = policy_map.get(tool)
    if not rules:
        return "deny"
    fallback = FALLBACK_ERROR
    for rule in progent_sorted([_as_rule(r) for r in rules]):
        fallback = rule.fallback
        if rule.effect == PROGENT_ALLOW:
            flag = True
            for arg, restriction in rule.condition.items():
                if arg in kwargs:
                    try:
                        _ref_check_arg(kwargs[arg], restriction)
                    except _RefInvalid:
                        flag = False
                        if (
                            rule.priority == HARD_PRIORITY
                            and rule.fallback == FALLBACK_ERROR
                        ):
                            return "deny"
                        break
            if flag:
                return "allow"
        else:
            try:
                for arg, restriction in rule.condition.items():
                    if arg in kwargs:
                        _ref_check_arg(kwargs[arg], restriction)
            except _RefInvalid:
                continue
            return "review" if rule.fallback == FALLBACK_CONFIRM else "deny"
    return "review" if fallback == FALLBACK_CONFIRM else "deny"


def _as_rule(raw) -> ProgentRule:
    if isinstance(raw, ProgentRule):
        return raw
    priority, effect, condition, fallback = raw
    return ProgentRule(
        priority=priority, effect=effect, condition=condition, fallback=fallback
    )


_FIDELITY_POLICIES: list[dict] = [
    # The banking flagship: forbid US, allow one IBAN at two amounts.
    {
        "send_money": [
            [0, 1, {"recipient": "US.*"}, 0],
            [1, 0, {"recipient": {"pattern": "^UK1$"}, "amount": {"enum": [10, 100]}}, 0],
        ],
        "get_balance": [[0, 0, {}, 0]],
    },
    # Equal priority: forbid runs before allow.
    {"t": [[1, 0, {"a": {"const": "x"}}, 0], [1, 1, {"a": {"pattern": "x"}}, 0]]},
    # Confirm fallbacks, firing and leaked.
    {"t": [[0, 1, {"a": {"const": "bad"}}, 2], [1, 0, {"a": {"const": "ok"}}, 2]]},
    # Hard allow with a later (dead) allow.
    {
        "t": [
            [100, 0, {"a": {"enum": ["x", "y"]}}, 0],
            [101, 0, {"a": {"const": "z"}}, 0],
        ]
    },
    # Terminate fallback and a type-string gate.
    {"t": [[0, 1, {"a": {"type": "string", "pattern": "^sec"}}, 1], [1, 0, {}, 0]]},
]

_FIDELITY_CALLS: list[tuple[str, dict]] = [
    ("send_money", {"recipient": "US99", "amount": 10}),
    ("send_money", {"recipient": "UK1", "amount": 10}),
    ("send_money", {"recipient": "UK1", "amount": 100}),
    ("send_money", {"recipient": "UK1", "amount": 11}),
    ("send_money", {"recipient": "DE7", "amount": 10}),
    ("get_balance", {}),
    ("unknown_tool", {}),
    ("t", {"a": "x"}),
    ("t", {"a": "y"}),
    ("t", {"a": "z"}),
    ("t", {"a": "ok"}),
    ("t", {"a": "bad"}),
    ("t", {"a": "secret"}),
    ("t", {"a": 42}),
]

_VERDICT_OF_ACTION = {"allow": "allow", "deny": "deny", "review": "review"}


class TestFidelity:
    @pytest.mark.parametrize("policy_map", _FIDELITY_POLICIES)
    @pytest.mark.parametrize("call", _FIDELITY_CALLS)
    def test_converted_policy_agrees_with_reference(
        self, policy_map: dict, call: tuple[str, dict]
    ) -> None:
        tool, kwargs = call
        expected = _reference_decide(policy_map, tool, kwargs)
        policy = _convert(policy_map)
        action = _action(policy, tool, kwargs)
        assert action is not None, "standalone conversion always decides"
        assert _VERDICT_OF_ACTION[action] == expected, (tool, kwargs)

    def test_documented_divergence_absent_argument(self) -> None:
        # Progent checks a restriction only when the argument is present:
        # the reference allows a call that omits the constrained argument,
        # the (stricter, fail-closed) translation denies it.
        policy_map = {"t": [[0, 0, {"a": {"const": "x"}}, 0]]}
        assert _reference_decide(policy_map, "t", {}) == "allow"
        assert _action(_convert(policy_map), "t", {}) == "deny"


# ----- CLI: apg policy import-progent ---------------------------------------


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture()
def rules_file(tmp_path: Path) -> Path:
    p = tmp_path / "rules.json"
    p.write_text(
        json.dumps(
            {
                "send_money": [
                    [0, 1, {"recipient": "US.*"}, 0],
                    [1, 0, {"recipient": {"pattern": "^UK1$"}}, 0],
                ]
            }
        ),
        encoding="utf-8",
    )
    return p


class TestImportProgentCli:
    def test_stdout_yaml_is_loadable(self, rules_file: Path) -> None:
        rc, out, err = _run(["policy", "import-progent", str(rules_file)])
        assert rc == 0 and err == ""
        policy = load_policy_str(out)
        assert policy.name == "progent-import"
        assert any(r.id == "progent:default" for r in policy.rules)

    def test_name_flag(self, rules_file: Path) -> None:
        rc, out, _ = _run(
            ["policy", "import-progent", str(rules_file), "--name", "bank"]
        )
        assert rc == 0
        assert load_policy_str(out).name == "bank"

    def test_default_per_tool(self, rules_file: Path) -> None:
        rc, out, _ = _run(
            ["policy", "import-progent", str(rules_file), "--default", "per-tool"]
        )
        assert rc == 0
        assert all(r.id != "progent:default" for r in load_policy_str(out).rules)

    def test_output_file(self, rules_file: Path, tmp_path: Path) -> None:
        target = tmp_path / "out.yaml"
        rc, out, err = _run(
            ["policy", "import-progent", str(rules_file), "-o", str(target)]
        )
        assert rc == 0 and err == ""
        assert str(target) in out and "rule(s)" in out
        from agent_policy_gateway import load_policy

        assert load_policy(target).name == "progent-import"

    def test_missing_file_exits_2(self, tmp_path: Path) -> None:
        rc, out, err = _run(["policy", "import-progent", str(tmp_path / "no.json")])
        assert rc == 2 and "not found" in err and out == ""

    def test_unwritable_output_exits_2(self, rules_file: Path, tmp_path: Path) -> None:
        target = tmp_path / "no-such-dir" / "out.yaml"
        rc, _, err = _run(
            ["policy", "import-progent", str(rules_file), "-o", str(target)]
        )
        assert rc == 2 and "cannot write" in err

    def test_invalid_json_exits_1(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{", encoding="utf-8")
        rc, out, err = _run(["policy", "import-progent", str(p)])
        assert rc == 1 and "cannot import" in err and out == ""

    def test_unsupported_subset_exits_1(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text(
            json.dumps({"t": [[0, 0, {"a": {"minLength": 2}}, 0]]}), encoding="utf-8"
        )
        rc, _, err = _run(["policy", "import-progent", str(p)])
        assert rc == 1 and "minLength" in err

    def test_validate_accepts_generated_file(
        self, rules_file: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "out.yaml"
        assert _run(["policy", "import-progent", str(rules_file), "-o", str(target)])[0] == 0
        rc, out, _ = _run(["policy", "validate", str(target)])
        assert rc == 0 and "OK" in out


# ----- CLI: explain / lint know about arg_matches ---------------------------


class TestArgMatchesCli:
    def _policy_file(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "policy.yaml"
        p.write_text(body, encoding="utf-8")
        return p

    def test_explain_traces_arg_matches_rejection(self, tmp_path: Path) -> None:
        p = self._policy_file(
            tmp_path,
            """
version: 1
name: t
rules:
  - id: r1
    when:
      tool: send_money
      arg_matches: {recipient: "^UK"}
    effect: {action: allow}
""",
        )
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                str(p),
                "--tool",
                "send_money",
                "--arg",
                "recipient=US99",
            ]
        )
        assert rc == 0
        assert "does not match regex" in out
        rc, out, _ = _run(["policy", "explain", str(p), "--tool", "send_money"])
        assert "is missing" in out

    def test_explain_traces_non_string_argument(self, tmp_path: Path) -> None:
        p = self._policy_file(
            tmp_path,
            """
version: 1
name: t
rules:
  - id: r1
    when: {tool: t, arg_matches: {n: "^1"}}
    effect: {action: allow}
""",
        )
        rc, out, _ = _run(
            ["policy", "explain", str(p), "--tool", "t", "--arg", "n=12"]
        )
        assert rc == 0 and "not a string" in out

    def test_lint_w002_on_equals_matches_contradiction(self, tmp_path: Path) -> None:
        p = self._policy_file(
            tmp_path,
            """
version: 1
name: t
rules:
  - id: dead
    when:
      tool: t
      arg_equals: {a: "xyz"}
      arg_matches: {a: "^q"}
    effect: {action: deny}
""",
        )
        rc, out, _ = _run(["policy", "lint", str(p)])
        assert rc == 3 and "W002" in out and "dead" in out

    def test_lint_w002_on_non_string_equals_with_regex(self, tmp_path: Path) -> None:
        p = self._policy_file(
            tmp_path,
            """
version: 1
name: t
rules:
  - id: dead
    when:
      tool: t
      arg_equals: {a: 7}
      arg_matches: {a: "^7"}
    effect: {action: deny}
""",
        )
        rc, out, _ = _run(["policy", "lint", str(p)])
        assert rc == 3 and "only matches strings" in out

    def test_lint_w001_identical_arg_matches_shadow(self, tmp_path: Path) -> None:
        p = self._policy_file(
            tmp_path,
            """
version: 1
name: t
rules:
  - id: first
    when: {tool: t, arg_matches: {a: "^x"}}
    effect: {action: allow}
  - id: second
    when: {tool: t, arg_matches: {a: "^x"}}
    effect: {action: deny}
""",
        )
        rc, out, _ = _run(["policy", "lint", str(p)])
        assert rc == 3 and "W001" in out and "second" in out

    def test_lint_no_false_shadow_on_different_patterns(self, tmp_path: Path) -> None:
        p = self._policy_file(
            tmp_path,
            """
version: 1
name: t
rules:
  - id: first
    when: {tool: t, arg_matches: {a: "^x"}}
    effect: {action: allow}
  - id: second
    when: {tool: t, arg_matches: {a: "^y"}}
    effect: {action: deny}
""",
        )
        rc, out, _ = _run(["policy", "lint", str(p)])
        assert rc == 0

    def test_lint_generated_demo_policy_is_clean(self) -> None:
        # The converter should not generate policies that trip its own lint.
        from agent_policy_gateway.cli import _lint
        from examples.progent import build_policy

        assert _lint(build_policy()) == []


# ----- worked example -------------------------------------------------------


class TestExample:
    def test_run_demo_matches_expectations(self) -> None:
        from examples.progent import run_demo

        for description, expected, decision in run_demo():
            assert decision.verdict == expected, description

    def test_demo_denials_carry_progent_rule_ids(self) -> None:
        from examples.progent import run_demo

        by_desc = {d: dec for d, _, dec in run_demo()}
        assert (
            by_desc["transfer to a US recipient (forbid rule)"].rule_id
            == "progent:send_money:0"
        )
        assert by_desc["tool with no Progent entry"].rule_id == "progent:default"

    def test_entry_point_exits_zero(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "examples.progent"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "[ok ]" in proc.stdout and "FAIL" not in proc.stdout

    def test_shipped_rules_round_trip_through_the_cli(self, tmp_path: Path) -> None:
        target = tmp_path / "generated.yaml"
        rc, _, err = _run(
            [
                "policy",
                "import-progent",
                str(REPO_ROOT / "examples" / "progent" / "rules.json"),
                "--name",
                "progent-banking-demo",
                "-o",
                str(target),
            ]
        )
        assert rc == 0, err
        from agent_policy_gateway import load_policy
        from examples.progent import build_policy

        assert load_policy(target) == build_policy()


# ----- Verdict sanity -------------------------------------------------------


class TestGatewayIntegration:
    def test_review_reaches_the_gateway_verdict(self) -> None:
        from agent_policy_gateway import Gateway

        policy = _convert({"t": [[0, 1, {}, FALLBACK_CONFIRM]]})
        gw = Gateway(policies=[policy])
        decision = gw.decide(ToolCall(tool_name="t", args={}))
        assert decision.verdict == Verdict.REVIEW
