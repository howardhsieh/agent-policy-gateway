"""Tests for dual-label taint: confidentiality + integrity (R51).

Covers:
* `TaintLabel` dimension algebra: canonical form (a source in both
  dimension sets is promoted to `sources`), semantic equality, the
  effective per-dimension / union properties, per-dimension `join` and
  `subsumes`, `without`, `of_dimensions`, `is_empty`.
* Serialization back-compat: a legacy label's `to_dict` shape is
  byte-for-byte unchanged; dimension keys appear only when present and
  round-trip through `from_dict` / `to_json`.
* Per-dimension `propagate`: dimension-scoped adds; confidentiality-only
  declassification leaving integrity taint intact (and the endorsement
  dual); the all-legacy case reducing exactly to the pre-R51 answer;
  provenance restriction on the union of dimensions.
* Policy DSL: nested `confidentiality:` / `integrity:` sub-conditions in
  YAML, matched against that dimension's effective set; top-level
  clauses matching the union; legacy `sources` counting in both
  dimensions; unknown fields rejected; `is_empty` accounting for
  dimension sub-conditions.
* Gateway end-to-end: an integrity-scoped rule refusing a privileged
  sink while a confidentiality-scoped rule refuses a public sink, each
  blind to the other dimension; redact declassify stripping a source
  from every dimension.
* CLI: `--taint conf:x` / `integ:x` prefixes; lint W002 for dimension
  contradictions; the conservative W001 shadow guard; explain's
  dimension-annotated taint rendering.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from agent_policy_gateway import (
    Decision,
    DimensionTaintCondition,
    Gateway,
    PolicyDenied,
    TaintCondition,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    load_policy_str,
    propagate,
    propagate_provenance,
)
from agent_policy_gateway.cli import _parse_taint, main
from agent_policy_gateway.core import Provenance, ProvenanceEntry
from agent_policy_gateway.policy import PolicyError

# --- label algebra ------------------------------------------------------------


class TestDualTaintLabel:
    def test_of_dimensions_constructor(self) -> None:
        lbl = TaintLabel.of_dimensions(
            both=["w"], confidentiality=["pii"], integrity=["web"]
        )
        assert lbl.sources == frozenset({"w"})
        assert lbl.confidentiality == frozenset({"pii"})
        assert lbl.integrity == frozenset({"web"})

    def test_canonical_form_promotes_source_in_both_dimensions(self) -> None:
        lbl = TaintLabel.of_dimensions(confidentiality=["x"], integrity=["x"])
        assert lbl.sources == frozenset({"x"})
        assert lbl.confidentiality == frozenset()
        assert lbl.integrity == frozenset()

    def test_canonical_form_drops_dimension_entry_already_in_sources(self) -> None:
        lbl = TaintLabel.of_dimensions(both=["x"], confidentiality=["x"])
        assert lbl == TaintLabel.of("x")

    def test_semantic_equality_across_representations(self) -> None:
        assert TaintLabel.of("x") == TaintLabel.of_dimensions(
            confidentiality=["x"], integrity=["x"]
        )

    def test_effective_dimension_properties(self) -> None:
        lbl = TaintLabel.of_dimensions(
            both=["w"], confidentiality=["pii"], integrity=["web"]
        )
        assert lbl.confidentiality_sources == frozenset({"w", "pii"})
        assert lbl.integrity_sources == frozenset({"w", "web"})
        assert lbl.all_sources == frozenset({"w", "pii", "web"})

    def test_join_is_per_dimension(self) -> None:
        a = TaintLabel.of_dimensions(confidentiality=["pii"])
        b = TaintLabel.of_dimensions(integrity=["web"])
        joined = a.join(b)
        assert joined.confidentiality == frozenset({"pii"})
        assert joined.integrity == frozenset({"web"})
        assert joined.sources == frozenset()

    def test_join_promotes_source_arriving_in_both_dimensions(self) -> None:
        a = TaintLabel.of_dimensions(confidentiality=["x"])
        b = TaintLabel.of_dimensions(integrity=["x"])
        assert a.join(b) == TaintLabel.of("x")

    def test_subsumes_is_per_dimension(self) -> None:
        both = TaintLabel.of("x")
        conf_only = TaintLabel.of_dimensions(confidentiality=["x"])
        assert both.subsumes(conf_only)
        assert not conf_only.subsumes(both)  # missing x in integrity

    def test_subsumes_legacy_degenerates_to_subset(self) -> None:
        assert TaintLabel.of("a", "b").subsumes(TaintLabel.of("a"))
        assert not TaintLabel.of("a").subsumes(TaintLabel.of("a", "b"))

    def test_without_strips_every_dimension(self) -> None:
        lbl = TaintLabel.of_dimensions(
            both=["x"], confidentiality=["x2"], integrity=["x3"]
        )
        assert lbl.without(["x", "x2", "x3"]).is_empty()
        assert lbl.without(["x2"]) == TaintLabel.of_dimensions(
            both=["x"], integrity=["x3"]
        )

    def test_is_empty_accounts_for_dimensions(self) -> None:
        assert TaintLabel().is_empty()
        assert not TaintLabel.of_dimensions(confidentiality=["pii"]).is_empty()
        assert not TaintLabel.of_dimensions(integrity=["web"]).is_empty()


class TestDualLabelSerialization:
    def test_legacy_label_dict_shape_unchanged(self) -> None:
        assert TaintLabel.of("b", "a").to_dict() == {"sources": ["a", "b"]}

    def test_dimension_keys_only_when_present(self) -> None:
        d = TaintLabel.of_dimensions(both=["w"], integrity=["web"]).to_dict()
        assert d == {"sources": ["w"], "integrity": ["web"]}
        assert "confidentiality" not in d

    def test_round_trip(self) -> None:
        lbl = TaintLabel.of_dimensions(
            both=["w"], confidentiality=["pii"], integrity=["web"]
        )
        assert TaintLabel.from_dict(lbl.to_dict()) == lbl

    def test_legacy_dict_loads(self) -> None:
        assert TaintLabel.from_dict({"sources": ["web"]}) == TaintLabel.of("web")

    def test_tool_call_round_trip_with_dimensions(self) -> None:
        call = ToolCall(
            tool_name="t",
            input_label=TaintLabel.of_dimensions(confidentiality=["pii"]),
        )
        assert ToolCall.from_dict(call.to_dict()) == call

    def test_decision_round_trip_with_dimensions(self) -> None:
        dec = Decision(
            verdict=Verdict.ALLOW,
            output_label=TaintLabel.of_dimensions(integrity=["web"]),
        )
        assert Decision.from_dict(dec.to_dict()) == dec

    def test_json_is_stable(self) -> None:
        lbl = TaintLabel.of_dimensions(both=["w"], confidentiality=["pii"])
        as_json = json.dumps(lbl.to_dict(), sort_keys=True)
        assert TaintLabel.from_dict(json.loads(as_json)) == lbl


# --- propagation --------------------------------------------------------------


class TestDualPropagation:
    def test_dimension_scoped_adds(self) -> None:
        spec = ToolTaintSpec.of(adds_integrity=["web"])
        out = propagate([TaintLabel()], spec)
        assert out == TaintLabel.of_dimensions(integrity=["web"])
        assert out.confidentiality_sources == frozenset()

    def test_declassify_confidentiality_leaves_integrity(self) -> None:
        # A legacy (both-dimensions) source declassified for
        # confidentiality only keeps its integrity taint: the redactor
        # removed the secret, not the untrustedness.
        spec = ToolTaintSpec.of(declassifies_confidentiality=["web"])
        out = propagate([TaintLabel.of("web")], spec)
        assert out == TaintLabel.of_dimensions(integrity=["web"])

    def test_endorse_leaves_confidentiality(self) -> None:
        spec = ToolTaintSpec.of(declassifies_integrity=["pii"])
        out = propagate([TaintLabel.of("pii")], spec)
        assert out == TaintLabel.of_dimensions(confidentiality=["pii"])

    def test_legacy_declassify_strips_both_dimensions(self) -> None:
        spec = ToolTaintSpec.of(declassifies=["web"])
        assert propagate([TaintLabel.of("web")], spec).is_empty()

    def test_all_legacy_reduces_to_pre_r51_answer(self) -> None:
        spec = ToolTaintSpec.of(adds=["tool"], declassifies=["pii"])
        out = propagate([TaintLabel.of("web", "pii")], spec)
        assert out == TaintLabel.of("web", "tool")
        assert out.confidentiality == frozenset()
        assert out.integrity == frozenset()

    def test_dimension_scoped_declassify_on_dimension_scoped_taint(self) -> None:
        label = TaintLabel.of_dimensions(confidentiality=["pii"], integrity=["web"])
        spec = ToolTaintSpec.of(declassifies_confidentiality=["pii"])
        assert propagate([label], spec) == TaintLabel.of_dimensions(integrity=["web"])

    def test_join_of_mixed_inputs(self) -> None:
        out = propagate(
            [
                TaintLabel.of("w"),
                TaintLabel.of_dimensions(confidentiality=["pii"]),
                TaintLabel.of_dimensions(integrity=["web"]),
            ]
        )
        assert out == TaintLabel.of_dimensions(
            both=["w"], confidentiality=["pii"], integrity=["web"]
        )

    def test_provenance_restricts_to_union_of_dimensions(self) -> None:
        # A conf-only surviving source must keep its provenance entry.
        spec = ToolTaintSpec.of(declassifies_integrity=["pii"])
        label = propagate([TaintLabel.of("pii")], spec)
        prov = propagate_provenance(
            [Provenance((ProvenanceEntry(source="pii", tool_name="crm"),))],
            spec,
            tool_name="t",
            output_label=label,
        )
        assert prov.origins("pii")

    def test_provenance_stamps_dimension_scoped_adds(self) -> None:
        spec = ToolTaintSpec.of(adds_integrity=["web"])
        label = propagate([TaintLabel()], spec)
        prov = propagate_provenance(
            [Provenance()], spec, tool_name="fetch", call_id="c1", output_label=label
        )
        assert prov.origins("web") == (
            ProvenanceEntry(source="web", tool_name="fetch", call_id="c1"),
        )


# --- policy DSL ---------------------------------------------------------------

DUAL_POLICY_YAML = """
version: 1
name: dual-label-demo
rules:
  - id: deny-untrusted-privileged
    when:
      tool: send_money
      taint:
        integrity:
          any_of: [web]
    effect:
      action: deny
      reason: untrusted data must not drive privileged actions
  - id: deny-secret-public
    when:
      tool: post_public
      taint:
        confidentiality:
          any_of: [pii]
    effect:
      action: deny
      reason: secret data must not reach public sinks
"""


class TestDimensionTaintConditionModel:
    def test_yaml_loads_dimension_sub_conditions(self) -> None:
        policy = load_policy_str(DUAL_POLICY_YAML)
        cond = policy.rules[0].when.taint
        assert cond is not None
        assert cond.integrity == DimensionTaintCondition(any_of=("web",))
        assert cond.confidentiality is None

    def test_unknown_field_in_dimension_condition_rejected(self) -> None:
        bad = DUAL_POLICY_YAML.replace("any_of: [web]", "sum_of: [web]")
        with pytest.raises(PolicyError):
            load_policy_str(bad)

    def test_is_empty_accounts_for_dimension_sub_conditions(self) -> None:
        assert TaintCondition().is_empty()
        assert TaintCondition(confidentiality=DimensionTaintCondition()).is_empty()
        assert not TaintCondition(
            integrity=DimensionTaintCondition(none_of=("web",))
        ).is_empty()

    def test_dimension_condition_sees_only_its_dimension(self) -> None:
        cond = TaintCondition(integrity=DimensionTaintCondition(any_of=("web",)))
        assert not cond.matches(TaintLabel.of_dimensions(confidentiality=["web"]))
        assert cond.matches(TaintLabel.of_dimensions(integrity=["web"]))

    def test_legacy_sources_count_in_both_dimensions(self) -> None:
        conf = TaintCondition(confidentiality=DimensionTaintCondition(any_of=("x",)))
        integ = TaintCondition(integrity=DimensionTaintCondition(any_of=("x",)))
        both = TaintLabel.of("x")
        assert conf.matches(both)
        assert integ.matches(both)

    def test_top_level_clauses_match_the_union(self) -> None:
        cond = TaintCondition(any_of=("web",))
        assert cond.matches(TaintLabel.of_dimensions(integrity=["web"]))
        assert cond.matches(TaintLabel.of_dimensions(confidentiality=["web"]))
        assert cond.matches(TaintLabel.of("web"))
        assert not cond.matches(TaintLabel())

    def test_top_level_none_of_forbids_in_every_dimension(self) -> None:
        cond = TaintCondition(none_of=("web",))
        assert not cond.matches(TaintLabel.of_dimensions(integrity=["web"]))
        assert cond.matches(TaintLabel.of("other"))

    def test_all_clauses_must_hold_together(self) -> None:
        cond = TaintCondition(
            any_of=("web",),
            confidentiality=DimensionTaintCondition(none_of=("pii",)),
        )
        assert cond.matches(TaintLabel.of_dimensions(integrity=["web"]))
        assert not cond.matches(
            TaintLabel.of_dimensions(integrity=["web"], confidentiality=["pii"])
        )


# --- gateway end-to-end -------------------------------------------------------


class TestGatewayDualLabels:
    def _gateway(self) -> Gateway:
        return Gateway(policies=[load_policy_str(DUAL_POLICY_YAML)])

    def test_untrusted_integrity_taint_denies_privileged_sink(self) -> None:
        gw = self._gateway()
        call = ToolCall(
            tool_name="send_money",
            input_label=TaintLabel.of_dimensions(integrity=["web"]),
        )
        with pytest.raises(PolicyDenied):
            gw.execute(call, lambda: "sent")

    def test_secret_data_is_free_to_drive_privileged_sink(self) -> None:
        # pii is confidentiality-scoped: it must not go public, but it is
        # not untrusted, so the integrity rule on send_money ignores it.
        gw = self._gateway()
        call = ToolCall(
            tool_name="send_money",
            input_label=TaintLabel.of_dimensions(confidentiality=["pii"]),
        )
        result, decision = gw.execute(call, lambda: "sent")
        assert result == "sent"
        assert decision.verdict == Verdict.ALLOW

    def test_secret_confidentiality_taint_denies_public_sink(self) -> None:
        gw = self._gateway()
        call = ToolCall(
            tool_name="post_public",
            input_label=TaintLabel.of_dimensions(confidentiality=["pii"]),
        )
        with pytest.raises(PolicyDenied):
            gw.execute(call, lambda: "posted")

    def test_untrusted_data_is_free_to_go_public(self) -> None:
        gw = self._gateway()
        call = ToolCall(
            tool_name="post_public",
            input_label=TaintLabel.of_dimensions(integrity=["web"]),
        )
        result, _ = gw.execute(call, lambda: "posted")
        assert result == "posted"

    def test_legacy_both_dimensions_source_trips_both_rules(self) -> None:
        gw = self._gateway()
        for tool, src in (("send_money", "web"), ("post_public", "pii")):
            call = ToolCall(tool_name=tool, input_label=TaintLabel.of(src))
            with pytest.raises(PolicyDenied):
                gw.execute(call, lambda: "x")

    def test_endorsing_spec_unlocks_privileged_sink(self) -> None:
        # A sanitizer that endorses `web` (integrity strip) lets its
        # output drive send_money even though the secrecy taint remains.
        gw = self._gateway()
        gw.register_tool("sanitize", ToolTaintSpec.of(declassifies_integrity=["web"]))
        sanitize_call = ToolCall(
            tool_name="sanitize", input_label=TaintLabel.of("web")
        )
        _, decision = gw.execute(sanitize_call, lambda: "clean")
        assert decision.output_label == TaintLabel.of_dimensions(
            confidentiality=["web"]
        )
        send = ToolCall(tool_name="send_money", input_label=decision.output_label)
        result, _ = gw.execute(send, lambda: "sent")
        assert result == "sent"

    def test_redact_declassify_strips_every_dimension(self) -> None:
        policy = load_policy_str(
            """
version: 1
name: redact-dual
rules:
  - id: redact-pii
    when:
      tool: send_email
    effect:
      action: redact
      redact:
        fields: [body]
        declassify: [pii]
"""
        )
        gw = Gateway(policies=[policy])
        call = ToolCall(
            tool_name="send_email",
            args={"body": "secret"},
            input_label=TaintLabel.of_dimensions(
                confidentiality=["pii"], integrity=["pii"]
            ),
        )
        _, decision = gw.execute(call, lambda body: body, body="secret")
        assert decision.verdict == Verdict.REDACT
        assert decision.output_label.is_empty()


# --- CLI ----------------------------------------------------------------------


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


class TestCliDualTaint:
    def test_parse_taint_dimension_prefixes(self) -> None:
        assert _parse_taint("web,conf:pii,integ:rss") == TaintLabel.of_dimensions(
            both=["web"], confidentiality=["pii"], integrity=["rss"]
        )

    def test_parse_taint_bare_sources_unchanged(self) -> None:
        assert _parse_taint("web,pii") == TaintLabel.of("web", "pii")

    def test_explain_matches_integrity_rule(self, tmp_path: Path) -> None:
        pol = tmp_path / "dual.yaml"
        pol.write_text(DUAL_POLICY_YAML, encoding="utf-8")
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                str(pol),
                "--tool",
                "send_money",
                "--taint",
                "integ:web",
            ]
        )
        assert rc == 0
        assert "[MATCH ] deny-untrusted-privileged" in out
        assert "integ=['web']" in out

    def test_explain_conf_taint_does_not_match_integrity_rule(
        self, tmp_path: Path
    ) -> None:
        pol = tmp_path / "dual.yaml"
        pol.write_text(DUAL_POLICY_YAML, encoding="utf-8")
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                str(pol),
                "--tool",
                "send_money",
                "--taint",
                "conf:web",
            ]
        )
        assert rc == 0
        assert "integrity(any_of=['web']" in out
        assert "[MATCH ]" not in out

    def test_lint_flags_dimension_contradiction(self, tmp_path: Path) -> None:
        pol = tmp_path / "bad.yaml"
        pol.write_text(
            """
version: 1
name: contradiction
rules:
  - id: never-matches
    when:
      taint:
        integrity:
          all_of: [web]
          none_of: [web]
    effect:
      action: deny
""",
            encoding="utf-8",
        )
        rc, out, _ = _run(["policy", "lint", str(pol)])
        assert rc == 3
        assert "W002" in out
        assert "integrity clause requires source 'web'" in out

    def test_lint_flags_top_level_none_of_vs_dimension_all_of(
        self, tmp_path: Path
    ) -> None:
        pol = tmp_path / "bad.yaml"
        pol.write_text(
            """
version: 1
name: contradiction
rules:
  - id: never-matches
    when:
      taint:
        none_of: [web]
        confidentiality:
          all_of: [web]
    effect:
      action: deny
""",
            encoding="utf-8",
        )
        rc, out, _ = _run(["policy", "lint", str(pol)])
        assert rc == 3
        assert "W002" in out

    def test_lint_dimension_rule_does_not_shadow(self, tmp_path: Path) -> None:
        # Conservative W001: a rule with dimension sub-conditions never
        # claims to shadow a later rule, even when clause texts coincide.
        pol = tmp_path / "ok.yaml"
        pol.write_text(
            """
version: 1
name: no-shadow
rules:
  - id: integrity-web
    when:
      taint:
        integrity:
          any_of: [web]
    effect:
      action: deny
  - id: any-web
    when:
      taint:
        any_of: [web]
    effect:
      action: deny
""",
            encoding="utf-8",
        )
        rc, out, _ = _run(["policy", "lint", str(pol)])
        assert rc == 0
        assert "W001" not in out

    def test_lint_legacy_rule_still_shadows(self, tmp_path: Path) -> None:
        pol = tmp_path / "shadow.yaml"
        pol.write_text(
            """
version: 1
name: shadow
rules:
  - id: broad
    when:
      taint:
        any_of: [web]
    effect:
      action: deny
  - id: narrow
    when:
      tool: send_email
      taint:
        any_of: [web]
    effect:
      action: allow
""",
            encoding="utf-8",
        )
        rc, out, _ = _run(["policy", "lint", str(pol)])
        assert rc == 3
        assert "W001" in out
