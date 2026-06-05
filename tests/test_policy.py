"""Tests for the declarative policy DSL (R3).

Coverage target:

* :class:`TaintCondition` clause semantics (all_of / any_of / none_of,
  empty condition matches everything, combined clauses).
* :class:`Selector` matching: tool glob, identity, resource glob, taint
  condition, empty selector matches everything, ``resource=None`` skips
  the resource constraint.
* :class:`Effect` validation: ``rate_limit`` requires
  ``limit_per_minute`` > 0; other actions reject ``limit_per_minute``.
* :class:`Rule` / :class:`Policy` validation: non-empty id, unique ids,
  unsupported version, unknown fields rejected, frozen models.
* Loader (``load_policy``, ``load_policy_str``): YAML errors, empty
  files, non-mapping top-level, validation failures all raise
  :class:`PolicyError` with a helpful message; round-trip via tmp_path.
* The three example policies under ``policies/`` parse cleanly and
  exhibit the expected first-match behaviour for representative calls.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_policy_gateway import (
    Action,
    Effect,
    Policy,
    PolicyError,
    RedactSpec,
    Rule,
    Selector,
    TaintCondition,
    TaintLabel,
    ToolCall,
    load_policy,
    load_policy_str,
)

# ---------------------------------------------------------------------------
# TaintCondition
# ---------------------------------------------------------------------------


class TestTaintCondition:
    def test_empty_condition_matches_anything(self) -> None:
        cond = TaintCondition()
        assert cond.is_empty()
        assert cond.matches(TaintLabel())
        assert cond.matches(TaintLabel.of("web", "pii"))

    def test_all_of_requires_every_source(self) -> None:
        cond = TaintCondition(all_of=("web", "pii"))
        assert cond.matches(TaintLabel.of("web", "pii"))
        assert cond.matches(TaintLabel.of("web", "pii", "extra"))
        assert not cond.matches(TaintLabel.of("web"))
        assert not cond.matches(TaintLabel.of("pii"))
        assert not cond.matches(TaintLabel())

    def test_any_of_requires_at_least_one(self) -> None:
        cond = TaintCondition(any_of=("web", "pii"))
        assert cond.matches(TaintLabel.of("web"))
        assert cond.matches(TaintLabel.of("pii"))
        assert cond.matches(TaintLabel.of("web", "extra"))
        assert not cond.matches(TaintLabel.of("extra"))
        assert not cond.matches(TaintLabel())

    def test_none_of_excludes_listed_sources(self) -> None:
        cond = TaintCondition(none_of=("web",))
        assert cond.matches(TaintLabel())
        assert cond.matches(TaintLabel.of("pii"))
        assert not cond.matches(TaintLabel.of("web"))
        assert not cond.matches(TaintLabel.of("web", "pii"))

    def test_combined_clauses_all_must_hold(self) -> None:
        cond = TaintCondition(any_of=("pii",), none_of=("trusted-redactor",))
        assert cond.matches(TaintLabel.of("pii"))
        assert not cond.matches(TaintLabel.of("pii", "trusted-redactor"))
        assert not cond.matches(TaintLabel.of("web"))


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def _call(
    tool_name: str = "send_email",
    *,
    agent_id: str | None = None,
    label: TaintLabel | None = None,
) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        agent_id=agent_id,
        input_label=label or TaintLabel(),
    )


class TestSelector:
    def test_empty_selector_matches_every_call(self) -> None:
        s = Selector()
        assert s.matches(_call("anything"))
        assert s.matches(_call("send_email", agent_id="a", label=TaintLabel.of("web")))

    def test_tool_exact_and_glob(self) -> None:
        assert Selector(tool="send_email").matches(_call("send_email"))
        assert not Selector(tool="send_email").matches(_call("send_sms"))
        assert Selector(tool="send_*").matches(_call("send_sms"))
        assert Selector(tool="send_*").matches(_call("send_email"))
        assert not Selector(tool="send_*").matches(_call("read_file"))

    def test_identity_must_match_exactly(self) -> None:
        s = Selector(identity="agent.research")
        assert s.matches(_call(agent_id="agent.research"))
        assert not s.matches(_call(agent_id="agent.other"))
        assert not s.matches(_call(agent_id=None))

    def test_resource_glob_requires_resource_argument(self) -> None:
        s = Selector(resource="https://*")
        assert s.matches(_call(), resource="https://example.com/x")
        assert not s.matches(_call(), resource="http://example.com/x")
        # selector says "resource matters", caller passed none -> no match
        assert not s.matches(_call())

    def test_taint_condition_evaluated(self) -> None:
        s = Selector(taint=TaintCondition(any_of=("web",)))
        assert s.matches(_call(label=TaintLabel.of("web")))
        assert not s.matches(_call(label=TaintLabel()))

    def test_all_fields_must_match_simultaneously(self) -> None:
        s = Selector(
            tool="http_*",
            identity="agent.research",
            taint=TaintCondition(any_of=("web",)),
        )
        good = _call("http_post", agent_id="agent.research", label=TaintLabel.of("web"))
        assert s.matches(good)
        # wrong tool
        assert not s.matches(_call("send_email", agent_id="agent.research",
                                   label=TaintLabel.of("web")))
        # wrong identity
        assert not s.matches(_call("http_post", agent_id="agent.other",
                                   label=TaintLabel.of("web")))
        # wrong taint
        assert not s.matches(_call("http_post", agent_id="agent.research"))


# ---------------------------------------------------------------------------
# Selector.arg_equals (R25)
# ---------------------------------------------------------------------------


def _args_call(tool_name: str = "post_message", **args: object) -> ToolCall:
    return ToolCall(tool_name=tool_name, args=dict(args))


class TestArgEquals:
    def test_literal_string_match(self) -> None:
        s = Selector(arg_equals={"channel": "#public"})
        assert s.matches(_args_call(channel="#public"))

    def test_literal_value_mismatch(self) -> None:
        s = Selector(arg_equals={"channel": "#public"})
        assert not s.matches(_args_call(channel="#random"))

    def test_missing_argument_does_not_match(self) -> None:
        s = Selector(arg_equals={"channel": "#public"})
        assert not s.matches(_args_call(body="hi"))
        assert not s.matches(_args_call())

    def test_int_and_bool_values(self) -> None:
        s = Selector(arg_equals={"count": 3, "dry_run": True})
        assert s.matches(_args_call(count=3, dry_run=True))
        assert not s.matches(_args_call(count=4, dry_run=True))
        assert not s.matches(_args_call(count=3, dry_run=False))

    def test_bool_int_comparison_is_type_strict(self) -> None:
        # Python says True == 1, but YAML true and 1 are distinct scalars.
        assert not Selector(arg_equals={"flag": True}).matches(_args_call(flag=1))
        assert not Selector(arg_equals={"n": 1}).matches(_args_call(n=True))
        assert not Selector(arg_equals={"n": 0}).matches(_args_call(n=False))

    def test_every_listed_argument_must_match(self) -> None:
        s = Selector(arg_equals={"channel": "#public", "dry_run": False})
        assert s.matches(_args_call(channel="#public", dry_run=False))
        assert not s.matches(_args_call(channel="#public", dry_run=True))
        assert not s.matches(_args_call(channel="#public"))

    def test_extra_call_arguments_are_ignored(self) -> None:
        s = Selector(arg_equals={"channel": "#public"})
        assert s.matches(_args_call(channel="#public", body="hi", n=7))

    def test_absent_and_empty_arg_equals_do_not_constrain(self) -> None:
        assert Selector().matches(_args_call(channel="#x"))
        assert Selector(arg_equals=None).matches(_args_call())
        assert Selector(arg_equals={}).matches(_args_call())

    def test_combines_with_other_clauses(self) -> None:
        s = Selector(tool="post_*", arg_equals={"channel": "#public"})
        assert s.matches(_args_call("post_message", channel="#public"))
        assert not s.matches(_args_call("send_email", channel="#public"))
        assert not s.matches(_args_call("post_message", channel="#priv"))

    def test_rejects_non_scalar_values(self) -> None:
        with pytest.raises(ValidationError):
            Selector(arg_equals={"x": [1, 2]})  # type: ignore[dict-item]
        with pytest.raises(ValidationError):
            Selector(arg_equals={"x": {"y": 1}})  # type: ignore[dict-item]
        with pytest.raises(ValidationError):
            Selector(arg_equals={"x": 1.5})  # type: ignore[dict-item]
        with pytest.raises(ValidationError):
            Selector(arg_equals={"x": None})  # type: ignore[dict-item]

    def test_rejects_blank_keys(self) -> None:
        with pytest.raises(ValidationError):
            Selector(arg_equals={"": "v"})
        with pytest.raises(ValidationError):
            Selector(arg_equals={"   ": "v"})

    def test_loads_from_yaml(self) -> None:
        pol = load_policy_str(
            textwrap.dedent(
                """\
                version: 1
                name: arg-demo
                rules:
                  - id: deny-public-channel
                    when:
                      tool: post_message
                      arg_equals: {channel: "#public", count: 2, dry: false}
                    effect: {action: deny}
                """
            )
        )
        sel = pol.rules[0].when
        assert sel.arg_equals == {"channel": "#public", "count": 2, "dry": False}
        assert pol.first_match(
            _args_call(channel="#public", count=2, dry=False)
        ) is pol.rules[0]
        assert pol.first_match(_args_call(channel="#public", count=2, dry=True)) is None

    def test_yaml_rejects_bad_value_types(self) -> None:
        with pytest.raises(PolicyError, match="arg_equals"):
            load_policy_str(
                textwrap.dedent(
                    """\
                    version: 1
                    name: bad
                    rules:
                      - id: r1
                        when:
                          arg_equals: {x: [1, 2]}
                        effect: {action: deny}
                    """
                )
            )


# ---------------------------------------------------------------------------
# Effect
# ---------------------------------------------------------------------------


class TestEffect:
    def test_allow_deny_review_accept_no_limit(self) -> None:
        for action in (Action.ALLOW, Action.DENY, Action.REVIEW):
            Effect(action=action)
            Effect(action=action, reason="because")

    def test_rate_limit_requires_positive_limit(self) -> None:
        Effect(action=Action.RATE_LIMIT, limit_per_minute=10)
        with pytest.raises(ValueError, match="rate_limit effect requires"):
            Effect(action=Action.RATE_LIMIT)
        with pytest.raises(ValueError, match="rate_limit effect requires"):
            Effect(action=Action.RATE_LIMIT, limit_per_minute=0)
        with pytest.raises(ValueError, match="rate_limit effect requires"):
            Effect(action=Action.RATE_LIMIT, limit_per_minute=-1)

    def test_non_rate_limit_must_not_set_limit(self) -> None:
        with pytest.raises(ValueError, match="only allowed for action=rate_limit"):
            Effect(action=Action.ALLOW, limit_per_minute=10)
        with pytest.raises(ValueError, match="only allowed for action=rate_limit"):
            Effect(action=Action.DENY, limit_per_minute=10)


# ---------------------------------------------------------------------------
# Rule + Policy validation
# ---------------------------------------------------------------------------


class TestPolicyValidation:
    def test_minimal_valid_policy(self) -> None:
        policy = Policy(
            name="p",
            rules=(Rule(id="r1", effect=Effect(action=Action.ALLOW)),),
        )
        assert policy.version == 1
        assert policy.name == "p"
        assert len(policy.rules) == 1

    def test_rule_id_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError):
            Rule(id="", effect=Effect(action=Action.ALLOW))
        with pytest.raises(ValueError):
            Rule(id="   ", effect=Effect(action=Action.ALLOW))

    def test_policy_name_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError):
            Policy(name="")

    def test_unsupported_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported policy version"):
            Policy(name="p", version=2)

    def test_duplicate_rule_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate rule id"):
            Policy(
                name="p",
                rules=(
                    Rule(id="dup", effect=Effect(action=Action.ALLOW)),
                    Rule(id="dup", effect=Effect(action=Action.DENY)),
                ),
            )

    def test_models_are_frozen(self) -> None:
        rule = Rule(id="r", effect=Effect(action=Action.ALLOW))
        with pytest.raises(ValidationError):
            rule.id = "other"  # type: ignore[misc]

    def test_unknown_fields_rejected(self) -> None:
        # extra="forbid" on every model
        with pytest.raises(ValueError):
            Selector.model_validate({"tool": "x", "bogus": 1})
        with pytest.raises(ValueError):
            TaintCondition.model_validate({"any_of": ["a"], "bogus": 1})
        with pytest.raises(ValueError):
            Effect.model_validate({"action": "allow", "bogus": 1})


# ---------------------------------------------------------------------------
# first_match ordering
# ---------------------------------------------------------------------------


class TestFirstMatch:
    def test_first_matching_rule_wins(self) -> None:
        policy = Policy(
            name="p",
            rules=(
                Rule(
                    id="allow-research",
                    when=Selector(tool="publish_*", identity="agent.research"),
                    effect=Effect(action=Action.ALLOW),
                ),
                Rule(
                    id="deny-others",
                    when=Selector(tool="publish_*"),
                    effect=Effect(action=Action.DENY),
                ),
            ),
        )
        assert policy.first_match(
            ToolCall(tool_name="publish_blog", agent_id="agent.research"),
        ).id == "allow-research"
        assert policy.first_match(
            ToolCall(tool_name="publish_blog", agent_id="agent.other"),
        ).id == "deny-others"

    def test_no_rule_matches_returns_none(self) -> None:
        policy = Policy(
            name="p",
            rules=(
                Rule(
                    id="x",
                    when=Selector(tool="send_email"),
                    effect=Effect(action=Action.DENY),
                ),
            ),
        )
        assert policy.first_match(ToolCall(tool_name="other_tool")) is None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


VALID_YAML = textwrap.dedent(
    """
    version: 1
    name: t
    rules:
      - id: r1
        when:
          tool: send_email
          taint:
            any_of: [web]
        effect:
          action: deny
          reason: nope
    """
)


class TestLoader:
    def test_load_string(self) -> None:
        p = load_policy_str(VALID_YAML)
        assert p.name == "t"
        assert len(p.rules) == 1
        assert p.rules[0].effect.action == Action.DENY

    def test_load_file(self, tmp_path: Path) -> None:
        f = tmp_path / "p.yaml"
        f.write_text(VALID_YAML, encoding="utf-8")
        p = load_policy(f)
        assert p.rules[0].id == "r1"

    def test_invalid_yaml_raises_policy_error(self) -> None:
        with pytest.raises(PolicyError, match="invalid YAML"):
            load_policy_str("name: [unclosed")

    def test_empty_file_raises_policy_error(self) -> None:
        with pytest.raises(PolicyError, match="empty"):
            load_policy_str("")
        with pytest.raises(PolicyError, match="empty"):
            load_policy_str("   \n# only a comment\n")

    def test_top_level_must_be_mapping(self) -> None:
        with pytest.raises(PolicyError, match="top-level YAML must be a mapping"):
            load_policy_str("- 1\n- 2\n")
        with pytest.raises(PolicyError, match="top-level YAML must be a mapping"):
            load_policy_str("just a string")

    def test_validation_failures_become_policy_error(self) -> None:
        with pytest.raises(PolicyError):
            # missing required `name`
            load_policy_str("version: 1\nrules: []\n")
        with pytest.raises(PolicyError):
            # bad effect (rate_limit without limit)
            load_policy_str(
                textwrap.dedent(
                    """
                    version: 1
                    name: t
                    rules:
                      - id: r1
                        effect:
                          action: rate_limit
                    """
                )
            )

    def test_error_message_includes_source_label(self) -> None:
        with pytest.raises(PolicyError, match="my-source"):
            load_policy_str("- bad", source="my-source")


# ---------------------------------------------------------------------------
# Example policies under policies/
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICIES_DIR = REPO_ROOT / "policies"


def test_policies_directory_has_four_examples() -> None:
    yamls = sorted(POLICIES_DIR.glob("*.yaml"))
    names = {p.name for p in yamls}
    assert names == {
        "default.yaml",
        "redact-pii.yaml",
        "research-agent.yaml",
        "strict-pii.yaml",
    }


def test_every_example_policy_loads_clean() -> None:
    for path in sorted(POLICIES_DIR.glob("*.yaml")):
        policy = load_policy(path)
        assert isinstance(policy, Policy)
        assert policy.name
        assert policy.rules, f"{path.name} has no rules"


class TestDefaultPolicy:
    @pytest.fixture()
    def policy(self) -> Policy:
        return load_policy(POLICIES_DIR / "default.yaml")

    def test_web_tainted_email_is_denied(self, policy: Policy) -> None:
        call = ToolCall(tool_name="send_email", input_label=TaintLabel.of("web"))
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "deny-web-to-email"
        assert rule.effect.action == Action.DENY

    def test_pii_http_post_is_reviewed(self, policy: Policy) -> None:
        call = ToolCall(tool_name="http_post", input_label=TaintLabel.of("pii"))
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "review-pii-egress"
        assert rule.effect.action == Action.REVIEW

    def test_kb_lookup_is_allowed(self, policy: Policy) -> None:
        call = ToolCall(tool_name="kb_lookup")
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "allow-internal-readers"
        assert rule.effect.action == Action.ALLOW

    def test_unmentioned_tool_falls_through(self, policy: Policy) -> None:
        call = ToolCall(tool_name="unrelated_tool")
        assert policy.first_match(call) is None


class TestResearchAgentPolicy:
    @pytest.fixture()
    def policy(self) -> Policy:
        return load_policy(POLICIES_DIR / "research-agent.yaml")

    def test_web_search_is_rate_limited_for_research_agent(self, policy: Policy) -> None:
        call = ToolCall(tool_name="web_search", agent_id="agent.research")
        rule = policy.first_match(call)
        assert rule is not None and rule.effect.action == Action.RATE_LIMIT
        assert rule.effect.limit_per_minute == 30

    def test_external_post_is_reviewed(self, policy: Policy) -> None:
        call = ToolCall(tool_name="http_post")
        rule = policy.first_match(call, resource="https://example.com/x")
        assert rule is not None and rule.id == "review-external-post"

    def test_publish_allowed_for_research_agent(self, policy: Policy) -> None:
        call = ToolCall(tool_name="publish_blog", agent_id="agent.research")
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "allow-research-publish"

    def test_publish_denied_for_other_agents(self, policy: Policy) -> None:
        call = ToolCall(tool_name="publish_blog", agent_id="agent.other")
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "deny-publish-from-other-agents"


class TestStrictPiiPolicy:
    @pytest.fixture()
    def policy(self) -> Policy:
        return load_policy(POLICIES_DIR / "strict-pii.yaml")

    def test_pii_http_post_denied_without_redactor(self, policy: Policy) -> None:
        call = ToolCall(tool_name="http_post", input_label=TaintLabel.of("pii"))
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "deny-pii-to-external"

    def test_redacted_pii_http_post_falls_through(self, policy: Policy) -> None:
        call = ToolCall(
            tool_name="http_post",
            input_label=TaintLabel.of("pii", "trusted-redactor"),
        )
        # All deny rules require none_of=[trusted-redactor]; storage_write rule
        # is the only review one and it's tool-specific. So this should fall
        # through to None.
        assert policy.first_match(call) is None

    def test_pii_email_denied(self, policy: Policy) -> None:
        call = ToolCall(tool_name="send_email", input_label=TaintLabel.of("pii"))
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "deny-pii-to-email"

    def test_pii_storage_write_reviewed(self, policy: Policy) -> None:
        call = ToolCall(tool_name="storage_write", input_label=TaintLabel.of("pii"))
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "review-pii-to-storage"
        assert rule.effect.action == Action.REVIEW


# ---------------------------------------------------------------------------
# RedactSpec + redact effect validation (R17)
# ---------------------------------------------------------------------------


class TestRedactSpec:
    def test_whole_value_mask_when_no_pattern(self) -> None:
        spec = RedactSpec(fields=("body",))
        assert spec.redact_value("secret@example.com") == "[REDACTED]"
        # Non-string values are masked too when there is no pattern.
        assert spec.redact_value({"k": "v"}) == "[REDACTED]"

    def test_pattern_masks_only_matching_substrings(self) -> None:
        spec = RedactSpec(
            fields=("body",),
            pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            mask="[EMAIL]",
        )
        out = spec.redact_value("ping bob@example.com now")
        assert out == "ping [EMAIL] now"

    def test_pattern_leaves_non_strings_untouched(self) -> None:
        spec = RedactSpec(fields=("body",), pattern=r"\d+")
        sentinel = {"k": 1}
        assert spec.redact_value(sentinel) is sentinel

    def test_empty_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedactSpec(fields=())

    def test_blank_field_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedactSpec(fields=("   ",))

    def test_invalid_pattern_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedactSpec(fields=("body",), pattern="(")


class TestRedactEffectShape:
    def test_redact_effect_requires_redact_block(self) -> None:
        with pytest.raises(ValidationError):
            Effect(action=Action.REDACT)

    def test_redact_block_forbidden_for_other_actions(self) -> None:
        with pytest.raises(ValidationError):
            Effect(action=Action.ALLOW, redact=RedactSpec(fields=("body",)))

    def test_valid_redact_effect(self) -> None:
        eff = Effect(
            action=Action.REDACT,
            redact=RedactSpec(fields=("body",), declassify=("pii",)),
        )
        assert eff.action == Action.REDACT
        assert eff.redact is not None
        assert eff.redact.fields == ("body",)

    def test_redact_effect_loads_from_yaml(self) -> None:
        text = textwrap.dedent(
            """
            version: 1
            name: redact-test
            rules:
              - id: mask-body
                when:
                  tool: send_email
                  taint:
                    any_of: [pii]
                effect:
                  action: redact
                  redact:
                    fields: [body]
                    mask: "[X]"
                    declassify: [pii]
                    add_label: [trusted-redactor]
            """
        )
        pol = load_policy_str(text)
        rule = pol.rules[0]
        assert rule.effect.action == Action.REDACT
        assert rule.effect.redact.mask == "[X]"
        assert rule.effect.redact.declassify == ("pii",)

    def test_redact_rule_with_unknown_redact_key_rejected(self) -> None:
        text = textwrap.dedent(
            """
            version: 1
            name: bad
            rules:
              - id: r
                effect:
                  action: redact
                  redact:
                    fields: [body]
                    bogus: 1
            """
        )
        with pytest.raises(PolicyError):
            load_policy_str(text)


class TestRedactPiiPolicy:
    @pytest.fixture()
    def policy(self) -> Policy:
        return load_policy(POLICIES_DIR / "redact-pii.yaml")

    def test_email_pii_matches_redact_rule(self, policy: Policy) -> None:
        call = ToolCall(tool_name="send_email", input_label=TaintLabel.of("pii"))
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "redact-pii-in-email"
        assert rule.effect.action == Action.REDACT

    def test_already_redacted_email_falls_through(self, policy: Policy) -> None:
        call = ToolCall(
            tool_name="send_email",
            input_label=TaintLabel.of("pii", "trusted-redactor"),
        )
        assert policy.first_match(call) is None

    def test_log_event_pii_matches_redact_rule(self, policy: Policy) -> None:
        call = ToolCall(tool_name="log_event", input_label=TaintLabel.of("pii"))
        rule = policy.first_match(call)
        assert rule is not None and rule.id == "redact-pii-in-log"
