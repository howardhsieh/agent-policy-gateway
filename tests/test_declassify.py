"""Tests for declarative declassify (R52).

Covers the ``DeclassifyGrant`` schema and matching, the governed-vs-
ungoverned gateway semantics (per-spec declassifies are inert once any
loaded policy declares grants), decision/audit serialization of fired
grant ids, the shipped ``policies/declassify-sanitizer.yaml`` example,
and the CLI surface (validate, explain grant trace, lint W002/W003).
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from agent_policy_gateway.audit import AuditRecord, format_record
from agent_policy_gateway.cli import main
from agent_policy_gateway.core import Decision, TaintLabel, ToolCall, Verdict
from agent_policy_gateway.gateway import Gateway, PolicyDenied
from agent_policy_gateway.policy import (
    DIMENSIONS,
    DeclassifyGrant,
    Policy,
    PolicyError,
    TaintCondition,
    load_policy,
    load_policy_str,
)
from agent_policy_gateway.taint import ToolTaintSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
SANITIZER_POLICY = REPO_ROOT / "policies" / "declassify-sanitizer.yaml"


def _grant(**overrides: object) -> DeclassifyGrant:
    base: dict[str, object] = {"id": "g1", "tool": "sanitize", "sources": ("web",)}
    base.update(overrides)
    return DeclassifyGrant.model_validate(base)


def _policy(*grants: DeclassifyGrant, rules: tuple = ()) -> Policy:
    return Policy(name="p", rules=rules, declassify=tuple(grants))


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestGrantSchema:
    def test_minimal_grant_defaults(self) -> None:
        g = _grant()
        assert g.dimensions == DIMENSIONS
        assert g.identity is None
        assert g.resource is None
        assert g.when is None
        assert g.description == ""

    def test_blank_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            _grant(id="  ")

    def test_blank_tool_rejected(self) -> None:
        with pytest.raises(ValueError):
            _grant(tool="")

    def test_empty_sources_rejected(self) -> None:
        with pytest.raises(ValueError):
            _grant(sources=())

    def test_blank_source_entry_rejected(self) -> None:
        with pytest.raises(ValueError):
            _grant(sources=("web", " "))

    def test_unknown_dimension_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown declassify dimension"):
            _grant(dimensions=("secrecy",))

    def test_empty_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError):
            _grant(dimensions=())

    def test_duplicate_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            _grant(dimensions=("integrity", "integrity"))

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValueError):
            _grant(bogus="x")

    def test_duplicate_grant_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate declassify grant id"):
            Policy(name="p", declassify=(_grant(), _grant()))

    def test_grant_id_may_equal_rule_id(self) -> None:
        # Rules and grants are separate namespaces.
        policy = load_policy_str(
            """
            version: 1
            name: p
            declassify:
              - id: same
                tool: t
                sources: [web]
            rules:
              - id: same
                effect: {action: deny}
            """
        )
        assert policy.rules[0].id == policy.declassify[0].id == "same"

    def test_yaml_round_trip(self) -> None:
        policy = load_policy_str(
            """
            version: 1
            name: p
            declassify:
              - id: g
                tool: sanitize_*
                identity: agent.a
                resource: "https://*"
                sources: [web, "rss.*"]
                dimensions: [integrity]
                when:
                  none_of: [pii]
            """
        )
        g = policy.declassify[0]
        assert g.tool == "sanitize_*"
        assert g.identity == "agent.a"
        assert g.sources == ("web", "rss.*")
        assert g.dimensions == ("integrity",)
        assert g.when is not None and g.when.none_of == ("pii",)

    def test_invalid_grant_in_yaml_is_policy_error(self) -> None:
        with pytest.raises(PolicyError):
            load_policy_str(
                """
                version: 1
                name: p
                declassify:
                  - id: g
                    tool: t
                    sources: []
                """
            )

    def test_policy_without_declassify_is_empty_tuple(self) -> None:
        assert Policy(name="p").declassify == ()


class TestGrantMatching:
    def test_tool_glob(self) -> None:
        g = _grant(tool="sanitize_*")
        assert g.matches(ToolCall(tool_name="sanitize_html"))
        assert not g.matches(ToolCall(tool_name="send_email"))

    def test_identity_condition(self) -> None:
        g = _grant(identity="agent.a")
        assert g.matches(ToolCall(tool_name="sanitize", agent_id="agent.a"))
        assert not g.matches(ToolCall(tool_name="sanitize", agent_id="agent.b"))
        assert not g.matches(ToolCall(tool_name="sanitize"))

    def test_resource_condition(self) -> None:
        g = _grant(resource="https://*")
        call = ToolCall(tool_name="sanitize")
        assert g.matches(call, resource="https://x.example")
        assert not g.matches(call, resource="ftp://x.example")
        # A resource constraint with no runtime resource never matches.
        assert not g.matches(call)

    def test_when_condition_on_input_label(self) -> None:
        g = _grant(when=TaintCondition(none_of=("pii",)))
        clean = ToolCall(tool_name="sanitize", input_label=TaintLabel.of("web"))
        dirty = ToolCall(
            tool_name="sanitize", input_label=TaintLabel.of("web", "pii")
        )
        assert g.matches(clean)
        assert not g.matches(dirty)

    def test_matching_grants_in_declaration_order(self) -> None:
        a, b, c = _grant(id="a"), _grant(id="b", tool="other"), _grant(id="c")
        policy = _policy(a, b, c)
        call = ToolCall(tool_name="sanitize")
        assert policy.matching_grants(call) == (a, c)


class TestGrantStrips:
    def test_literal_source(self) -> None:
        g = _grant(sources=("web",))
        assert g.strips(frozenset({"web", "pii"}), "integrity") == {"web"}

    def test_glob_source(self) -> None:
        g = _grant(sources=("crm.*",))
        srcs = frozenset({"crm.email", "crm.phone", "web"})
        assert g.strips(srcs, "confidentiality") == {"crm.email", "crm.phone"}

    def test_dimension_not_granted_is_empty(self) -> None:
        g = _grant(dimensions=("integrity",))
        assert g.strips(frozenset({"web"}), "confidentiality") == frozenset()


class TestUngovernedBackCompat:
    """With no grants anywhere, R52 changes nothing."""

    def test_spec_declassifies_still_apply(self) -> None:
        gw = Gateway(policies=[Policy(name="p")])
        gw.register_tool("redact", ToolTaintSpec.of(declassifies=("pii",)))
        call = ToolCall(tool_name="redact", input_label=TaintLabel.of("pii", "web"))
        decision = gw.decide(call)
        assert decision.output_label == TaintLabel.of("web")
        assert decision.declassified_by == ()

    def test_decision_dict_shape_unchanged(self) -> None:
        gw = Gateway()
        decision = gw.decide(ToolCall(tool_name="t"))
        assert "declassified_by" not in decision.to_dict()


def _governed_gateway(*grants: DeclassifyGrant, rules: tuple = ()) -> Gateway:
    return Gateway(policies=[_policy(*grants, rules=rules)])


class TestGovernedGateway:
    def test_spec_declassifies_inert_under_governance(self) -> None:
        gw = _governed_gateway(_grant(tool="unrelated"))
        gw.register_tool("redact", ToolTaintSpec.of(declassifies=("pii",)))
        call = ToolCall(tool_name="redact", input_label=TaintLabel.of("pii"))
        decision = gw.decide(call)
        # The spec's ad-hoc strip no longer applies: the policy is the
        # sole authority and no grant covers this tool.
        assert decision.output_label == TaintLabel.of("pii")
        assert decision.declassified_by == ()

    def test_grant_strips_both_dimensions(self) -> None:
        gw = _governed_gateway(_grant(sources=("web",)))
        call = ToolCall(tool_name="sanitize", input_label=TaintLabel.of("web", "x"))
        decision = gw.decide(call)
        assert decision.output_label == TaintLabel.of("x")
        assert decision.declassified_by == ("g1",)

    def test_integrity_only_strip_is_endorsement(self) -> None:
        gw = _governed_gateway(_grant(dimensions=("integrity",)))
        call = ToolCall(tool_name="sanitize", input_label=TaintLabel.of("web"))
        out = gw.decide(call).output_label
        # The untrustedness is endorsed away; the secrecy stays.
        assert out.confidentiality_sources == {"web"}
        assert out.integrity_sources == frozenset()

    def test_confidentiality_only_strip_keeps_integrity(self) -> None:
        gw = _governed_gateway(
            _grant(tool="redact", sources=("pii",), dimensions=("confidentiality",))
        )
        call = ToolCall(tool_name="redact", input_label=TaintLabel.of("pii"))
        out = gw.decide(call).output_label
        assert out.confidentiality_sources == frozenset()
        assert out.integrity_sources == {"pii"}

    def test_spec_adds_still_apply_and_can_be_stripped(self) -> None:
        gw = _governed_gateway(_grant(tool="fetch", sources=("web",)))
        gw.register_tool("fetch", ToolTaintSpec.of(adds=("web", "net")))
        decision = gw.decide(ToolCall(tool_name="fetch"))
        assert decision.output_label == TaintLabel.of("net")
        assert decision.declassified_by == ("g1",)

    def test_matching_grant_that_strips_nothing_not_recorded(self) -> None:
        gw = _governed_gateway(_grant(sources=("absent",)))
        call = ToolCall(tool_name="sanitize", input_label=TaintLabel.of("web"))
        decision = gw.decide(call)
        assert decision.output_label == TaintLabel.of("web")
        assert decision.declassified_by == ()

    def test_when_condition_gates_the_strip(self) -> None:
        gw = _governed_gateway(_grant(when=TaintCondition(none_of=("pii",))))
        clean = ToolCall(tool_name="sanitize", input_label=TaintLabel.of("web"))
        dirty = ToolCall(
            tool_name="sanitize", input_label=TaintLabel.of("web", "pii")
        )
        assert gw.decide(clean).output_label == TaintLabel()
        assert gw.decide(dirty).output_label == TaintLabel.of("web", "pii")

    def test_resource_condition_via_execute(self) -> None:
        gw = _governed_gateway(_grant(resource="https://trusted.example/*"))
        call = ToolCall(tool_name="sanitize", input_label=TaintLabel.of("web"))
        _, ok = gw.execute(
            call, lambda: None, resource="https://trusted.example/page"
        )
        assert ok.output_label == TaintLabel()
        _, bad = gw.execute(call, lambda: None, resource="https://evil.example/")
        assert bad.output_label == TaintLabel.of("web")

    def test_multiple_grants_union_and_order(self) -> None:
        gw = _governed_gateway(
            _grant(id="a", sources=("web",)),
            _grant(id="b", sources=("pii",)),
        )
        call = ToolCall(
            tool_name="sanitize", input_label=TaintLabel.of("web", "pii", "x")
        )
        decision = gw.decide(call)
        assert decision.output_label == TaintLabel.of("x")
        assert decision.declassified_by == ("a", "b")

    def test_grants_in_second_policy_govern_the_whole_gateway(self) -> None:
        rule_policy = load_policy_str(
            """
            version: 1
            name: rules
            rules:
              - id: allow-all
                effect: {action: allow}
            """
        )
        grant_policy = _policy(_grant())
        gw = Gateway(policies=[rule_policy, grant_policy])
        gw.register_tool("other", ToolTaintSpec.of(declassifies=("web",)))
        # The spec strip is inert even though the grant lives in policy 2
        # and the matching rule in policy 1.
        decision = gw.decide(
            ToolCall(tool_name="other", input_label=TaintLabel.of("web"))
        )
        assert decision.rule_id == "allow-all"
        assert decision.output_label == TaintLabel.of("web")
        sanitized = gw.decide(
            ToolCall(tool_name="sanitize", input_label=TaintLabel.of("web"))
        )
        assert sanitized.output_label == TaintLabel()
        assert sanitized.declassified_by == ("g1",)

    def test_provenance_restricted_after_grant_strip(self) -> None:
        gw = _governed_gateway(_grant(tool="fetch", sources=("web",)))
        gw.track_provenance = True
        gw.register_tool("fetch", ToolTaintSpec.of(adds=("web", "net")))
        decision = gw.decide(ToolCall(tool_name="fetch", call_id="c1"))
        assert {e.source for e in decision.output_provenance.entries} == {"net"}

    def test_redact_declassify_still_applies_when_governed(self) -> None:
        policy = load_policy_str(
            """
            version: 1
            name: p
            declassify:
              - id: g
                tool: unrelated
                sources: [web]
            rules:
              - id: mask
                when: {tool: send}
                effect:
                  action: redact
                  redact:
                    fields: [body]
                    declassify: [pii]
            """
        )
        gw = Gateway(policies=[policy])
        call = ToolCall(
            tool_name="send",
            args={"body": "s"},
            input_label=TaintLabel.of("pii"),
        )
        decision = gw.decide(call)
        assert decision.verdict == Verdict.REDACT
        assert decision.output_label == TaintLabel()

    def test_denied_call_still_reports_fired_grants(self) -> None:
        policy = load_policy_str(
            """
            version: 1
            name: p
            declassify:
              - id: g
                tool: send
                sources: [web]
            rules:
              - id: deny-send
                when: {tool: send}
                effect: {action: deny}
            """
        )
        gw = Gateway(policies=[policy])
        call = ToolCall(tool_name="send", input_label=TaintLabel.of("web"))
        with pytest.raises(PolicyDenied) as exc:
            gw.execute(call, lambda: None)
        assert exc.value.decision.declassified_by == ("g",)


class TestDecisionSerialization:
    def test_round_trip(self) -> None:
        d = Decision(verdict=Verdict.ALLOW, declassified_by=("a", "b"))
        assert d.to_dict()["declassified_by"] == ["a", "b"]
        assert Decision.from_dict(d.to_dict()) == d

    def test_empty_omitted_and_legacy_default(self) -> None:
        d = Decision(verdict=Verdict.ALLOW)
        payload = d.to_dict()
        assert "declassified_by" not in payload
        assert Decision.from_dict(payload).declassified_by == ()


class TestReplayRendering:
    def test_declassify_line_present_when_fired(self) -> None:
        record = AuditRecord(
            ts="2026-08-28T00:00:00+00:00",
            call=ToolCall(tool_name="sanitize"),
            decision=Decision(verdict=Verdict.ALLOW, declassified_by=("g1",)),
        )
        assert "declassify: g1" in format_record(record)

    def test_no_line_when_absent(self) -> None:
        record = AuditRecord(
            ts="2026-08-28T00:00:00+00:00",
            call=ToolCall(tool_name="sanitize"),
            decision=Decision(verdict=Verdict.ALLOW),
        )
        assert "declassify:" not in format_record(record)


class TestSanitizerExamplePolicy:
    def test_validates(self) -> None:
        rc, out, _ = _run(["policy", "validate", str(SANITIZER_POLICY)])
        assert rc == 0
        assert "declassify-sanitizer" in out

    def test_end_to_end_sanitize_then_send(self) -> None:
        gw = Gateway(policies=[load_policy(SANITIZER_POLICY)])
        gw.register_tool("fetch_page", ToolTaintSpec.of(adds=("web",)))
        web_label = gw.decide(ToolCall(tool_name="fetch_page")).output_label
        assert web_label.integrity_sources == {"web"}

        # Unsanitized web content cannot drive send_email.
        with pytest.raises(PolicyDenied):
            gw.execute(
                ToolCall(tool_name="send_email", input_label=web_label),
                lambda: None,
            )

        # The sanitizer endorses it (integrity strip; secrecy kept) ...
        endorsed = gw.decide(
            ToolCall(tool_name="sanitize_html", input_label=web_label)
        )
        assert endorsed.declassified_by == ("sanitizer-endorses-web",)
        assert endorsed.output_label.integrity_sources == frozenset()
        assert endorsed.output_label.confidentiality_sources == {"web"}

        # ... after which send_email is allowed.
        _, decision = gw.execute(
            ToolCall(tool_name="send_email", input_label=endorsed.output_label),
            lambda: None,
        )
        assert decision.verdict == Verdict.ALLOW

    def test_redactor_refuses_to_launder_web_tainted_pii(self) -> None:
        gw = Gateway(policies=[load_policy(SANITIZER_POLICY)])
        clean = gw.decide(
            ToolCall(tool_name="pii_redactor", input_label=TaintLabel.of("pii"))
        )
        assert clean.output_label.confidentiality_sources == frozenset()
        tainted = gw.decide(
            ToolCall(
                tool_name="pii_redactor",
                input_label=TaintLabel.of("pii", "web"),
            )
        )
        assert "pii" in tainted.output_label.confidentiality_sources
        assert tainted.declassified_by == ()


class TestCliExplain:
    def test_grant_trace_rendered(self) -> None:
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                str(SANITIZER_POLICY),
                "--tool",
                "sanitize_html",
                "--taint",
                "web",
            ]
        )
        assert rc == 0
        assert "declassify grants (2):" in out
        assert "[MATCH ] sanitizer-endorses-web" in out
        assert "from integrity" in out
        assert "[ no  ] redactor-declassifies-pii" in out

    def test_when_rejection_named(self) -> None:
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                str(SANITIZER_POLICY),
                "--tool",
                "pii_redactor",
                "--taint",
                "pii,web",
            ]
        )
        assert rc == 0
        assert "fails the when: condition" in out

    def test_no_trace_for_grantless_policy(self) -> None:
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                str(REPO_ROOT / "policies" / "default.yaml"),
                "--tool",
                "anything",
            ]
        )
        assert rc == 0
        assert "declassify grants" not in out


class TestCliLint:
    def _lint(self, tmp_path: Path, body: str) -> tuple[int, str]:
        path = tmp_path / "p.yaml"
        path.write_text(body, encoding="utf-8")
        rc, out, _ = _run(["policy", "lint", str(path)])
        return rc, out

    def test_sanitizer_policy_is_clean(self) -> None:
        rc, out, _ = _run(["policy", "lint", str(SANITIZER_POLICY)])
        assert rc == 0
        assert "no lint findings" in out

    def test_w003_unconditional_grant(self, tmp_path: Path) -> None:
        rc, out = self._lint(
            tmp_path,
            """
            version: 1
            name: p
            declassify:
              - id: strip-all
                tool: "*"
                sources: ["*"]
            """,
        )
        assert rc == 3
        assert "W003" in out and "strip-all" in out

    def test_conditioned_broad_grant_not_flagged(self, tmp_path: Path) -> None:
        rc, out = self._lint(
            tmp_path,
            """
            version: 1
            name: p
            declassify:
              - id: broad-but-endorse-only
                tool: "*"
                sources: ["*"]
                dimensions: [integrity]
            """,
        )
        assert rc == 0
        assert "W003" not in out

    def test_w002_contradictory_when(self, tmp_path: Path) -> None:
        rc, out = self._lint(
            tmp_path,
            """
            version: 1
            name: p
            declassify:
              - id: never
                tool: t
                sources: [web]
                when:
                  all_of: [x]
                  none_of: [x]
            """,
        )
        assert rc == 3
        assert "W002 declassify grant 'never' can never match" in out
