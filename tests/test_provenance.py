"""Tests for taint provenance chains (R19).

A :class:`TaintLabel` records *what* sources a value carries; the
:class:`Provenance` side-channel records *where each source came from*.
These tests cover the provenance value algebra, the
:func:`propagate_provenance` rule, and the headline acceptance scenario:
a 3-hop ``web_fetch -> summarize -> send_email`` flow run through a real
:class:`Gateway` produces an audit record on the denied send whose
provenance names the originating ``web_fetch`` call id.
"""

from __future__ import annotations

import json

import pytest

from agent_policy_gateway import (
    Decision,
    Gateway,
    PolicyDenied,
    Provenance,
    ProvenanceEntry,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    format_record,
    load_policy_str,
    propagate,
    propagate_provenance,
)
from agent_policy_gateway.audit import AuditRecord, JsonlAuditWriter

DENY_WEB_EMAIL = """
version: 1
name: deny-web-email
rules:
  - id: deny-web-to-email
    when:
      tool: send_email
      taint:
        any_of: [web]
    effect:
      action: deny
      reason: web-tainted content must not be emailed
"""


# --- Provenance value algebra -------------------------------------------------


class TestProvenanceValue:
    def test_empty_by_default(self) -> None:
        assert Provenance().is_empty()
        assert Provenance().entries == ()

    def test_add_appends_entry(self) -> None:
        p = Provenance().add(ProvenanceEntry("web", "web_fetch", "c1"))
        assert not p.is_empty()
        assert p.origins("web")[0].tool_name == "web_fetch"

    def test_dedupes_identical_entries(self) -> None:
        e = ProvenanceEntry("web", "web_fetch", "c1")
        p = Provenance((e, e)).add(e)
        assert len(p.entries) == 1

    def test_merge_is_union(self) -> None:
        a = Provenance().add(ProvenanceEntry("web", "web_fetch", "c1"))
        b = Provenance().add(ProvenanceEntry("crm", "crm_read", "c2"))
        merged = a.merge(b)
        assert {e.source for e in merged.entries} == {"web", "crm"}

    def test_origins_filters_by_source(self) -> None:
        p = (
            Provenance()
            .add(ProvenanceEntry("web", "web_fetch", "c1"))
            .add(ProvenanceEntry("crm", "crm_read", "c2"))
        )
        assert [e.tool_name for e in p.origins("web")] == ["web_fetch"]
        assert p.origins("missing") == ()

    def test_restrict_to_drops_declassified_sources(self) -> None:
        p = (
            Provenance()
            .add(ProvenanceEntry("web", "web_fetch", "c1"))
            .add(ProvenanceEntry("pii", "crm_read", "c2"))
        )
        kept = p.restrict_to({"web"})
        assert {e.source for e in kept.entries} == {"web"}

    def test_round_trip_dict(self) -> None:
        p = (
            Provenance()
            .add(ProvenanceEntry("web", "web_fetch", "c1"))
            .add(ProvenanceEntry("crm", "crm_read", None))
        )
        assert Provenance.from_dict(p.to_dict()) == p


# --- propagate_provenance rule ------------------------------------------------


class TestPropagateProvenance:
    def test_adds_stamp_originating_call(self) -> None:
        spec = ToolTaintSpec.of(adds=["web"])
        out_label = propagate([TaintLabel()], spec)
        prov = propagate_provenance(
            [Provenance()],
            spec,
            tool_name="web_fetch",
            call_id="c1",
            output_label=out_label,
        )
        entry = prov.origins("web")[0]
        assert (entry.tool_name, entry.call_id) == ("web_fetch", "c1")

    def test_transparent_tool_carries_provenance(self) -> None:
        upstream = Provenance().add(ProvenanceEntry("web", "web_fetch", "c1"))
        # summarize has no spec => transparent propagator
        prov = propagate_provenance(
            [upstream],
            None,
            tool_name="summarize",
            call_id="c2",
            output_label=TaintLabel.of("web"),
        )
        assert prov.origins("web")[0].call_id == "c1"

    def test_declassification_drops_provenance(self) -> None:
        upstream = Provenance().add(ProvenanceEntry("web", "web_fetch", "c1"))
        spec = ToolTaintSpec.of(declassifies=["web"])
        out_label = propagate([TaintLabel.of("web")], spec)
        assert out_label.is_empty()
        prov = propagate_provenance(
            [upstream],
            spec,
            tool_name="trusted_summary",
            call_id="c2",
            output_label=out_label,
        )
        assert prov.is_empty()


# --- gateway integration ------------------------------------------------------


class TestGatewayProvenance:
    def test_off_by_default_keeps_legacy_decision(self) -> None:
        gw = Gateway()
        gw.register_tool("web_fetch", ToolTaintSpec.of(adds=["web"]))
        decision = gw.decide(ToolCall(tool_name="web_fetch", call_id="c1"))
        assert decision.output_label == TaintLabel.of("web")
        assert decision.output_provenance.is_empty()

    def test_records_origin_when_enabled(self) -> None:
        gw = Gateway(track_provenance=True)
        gw.register_tool("web_fetch", ToolTaintSpec.of(adds=["web"]))
        decision = gw.decide(ToolCall(tool_name="web_fetch", call_id="c1"))
        assert decision.output_provenance.origins("web")[0].call_id == "c1"


# --- acceptance: 3-hop web -> summarize -> email ------------------------------


class TestThreeHopExfiltrationProvenance:
    """web_fetch -> summarize -> send_email; the denied send's audit record
    must name the originating web_fetch call id."""

    def test_denied_send_audit_record_names_web_fetch_origin(
        self, tmp_path
    ) -> None:
        log = tmp_path / "audit.jsonl"
        with JsonlAuditWriter(log) as writer:
            gw = Gateway(
                policies=[load_policy_str(DENY_WEB_EMAIL)],
                audit_writer=writer,
                track_provenance=True,
            )

            @gw.wrap_tool(
                tool_name="web_fetch", taint_spec=ToolTaintSpec.of(adds=["web"])
            )
            def web_fetch(url: str) -> str:
                return "untrusted page content"

            @gw.wrap_tool(tool_name="summarize")
            def summarize(text: str) -> str:
                return "summary: " + text

            @gw.wrap_tool(tool_name="send_email")
            def send_email(to: str, body: str) -> dict:  # pragma: no cover
                return {"sent": True}

            # Hop 1: web_fetch introduces the web taint.
            _, d1 = gw.execute(
                ToolCall(
                    tool_name="web_fetch",
                    args={"url": "http://evil.example"},
                    call_id="web-fetch-1",
                ),
                lambda: "untrusted page content",
            )
            assert d1.output_label == TaintLabel.of("web")

            # Hop 2: summarize carries the taint + provenance forward.
            _, d2 = gw.execute(
                ToolCall(
                    tool_name="summarize",
                    args={"text": "..."},
                    input_label=d1.output_label,
                    input_provenance=d1.output_provenance,
                    call_id="summarize-1",
                ),
                lambda: "summary",
            )
            assert d2.output_label == TaintLabel.of("web")

            # Hop 3: send_email is denied; the chain points back to web_fetch.
            with pytest.raises(PolicyDenied) as ei:
                gw.execute(
                    ToolCall(
                        tool_name="send_email",
                        args={"to": "x@y.z", "body": "..."},
                        input_label=d2.output_label,
                        input_provenance=d2.output_provenance,
                        call_id="send-email-1",
                    ),
                    lambda **_: {"sent": True},
                )
            denied = ei.value.decision
            assert denied.verdict is Verdict.DENY
            origin = denied.output_provenance.origins("web")[0]
            assert origin.tool_name == "web_fetch"
            assert origin.call_id == "web-fetch-1"

        # The audit log persists the provenance for the denied send.
        records = [
            AuditRecord.from_dict(json.loads(line))
            for line in log.read_text().splitlines()
            if line.strip()
        ]
        denied_records = [
            r for r in records if r.decision.verdict is Verdict.DENY
        ]
        assert len(denied_records) == 1
        rec = denied_records[0]
        assert rec.call.tool_name == "send_email"
        web_origin = rec.decision.output_provenance.origins("web")[0]
        assert web_origin.tool_name == "web_fetch"
        assert web_origin.call_id == "web-fetch-1"
        # And the human-readable timeline surfaces it.
        rendered = format_record(rec)
        assert "web<-web_fetch@web-fetch-1" in rendered


class TestDecisionProvenanceRoundTrip:
    def test_decision_serialization_includes_nonempty_provenance(self) -> None:
        prov = Provenance().add(ProvenanceEntry("web", "web_fetch", "c1"))
        d = Decision(
            verdict=Verdict.DENY,
            output_label=TaintLabel.of("web"),
            output_provenance=prov,
        )
        round_tripped = Decision.from_dict(d.to_dict())
        assert round_tripped == d
        assert "output_provenance" in d.to_dict()

    def test_empty_provenance_not_serialized(self) -> None:
        d = Decision(verdict=Verdict.ALLOW)
        assert "output_provenance" not in d.to_dict()
