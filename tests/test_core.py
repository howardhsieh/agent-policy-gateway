"""Tests for the core data model: TaintLabel, ToolCall, Decision, Verdict.

Covers construction, value equality, lattice algebra (join, subsumes), and
JSON / dict round-tripping. Intended to give >=90% coverage of core.py.
"""

from __future__ import annotations

import json

import pytest

from agent_policy_gateway.core import (
    Decision,
    TaintLabel,
    ToolCall,
    Verdict,
    from_json,
    to_json,
)

# --- TaintLabel ----------------------------------------------------------------


class TestTaintLabel:
    def test_default_is_empty(self) -> None:
        assert TaintLabel().sources == frozenset()
        assert TaintLabel().is_empty()

    def test_of_constructs_from_varargs(self) -> None:
        lbl = TaintLabel.of("web", "user_upload")
        assert lbl.sources == frozenset({"web", "user_upload"})

    def test_join_unions_sources(self) -> None:
        a = TaintLabel.of("web")
        b = TaintLabel.of("crm.contact.email")
        assert a.join(b).sources == {"web", "crm.contact.email"}

    def test_join_idempotent(self) -> None:
        a = TaintLabel.of("web")
        assert a.join(a) == a

    def test_join_commutative(self) -> None:
        a = TaintLabel.of("web")
        b = TaintLabel.of("crm")
        assert a.join(b) == b.join(a)

    def test_join_associative(self) -> None:
        a, b, c = TaintLabel.of("a"), TaintLabel.of("b"), TaintLabel.of("c")
        assert a.join(b).join(c) == a.join(b.join(c))

    def test_join_with_empty_is_identity(self) -> None:
        a = TaintLabel.of("web")
        assert a.join(TaintLabel()) == a
        assert TaintLabel().join(a) == a

    def test_subsumes_strict(self) -> None:
        web_and_crm = TaintLabel.of("web", "crm")
        web = TaintLabel.of("web")
        assert web_and_crm.subsumes(web)
        assert not web.subsumes(web_and_crm)

    def test_subsumes_self(self) -> None:
        a = TaintLabel.of("web")
        assert a.subsumes(a)

    def test_subsumes_empty(self) -> None:
        a = TaintLabel.of("web")
        assert a.subsumes(TaintLabel())
        assert TaintLabel().subsumes(TaintLabel())

    def test_value_equality(self) -> None:
        assert TaintLabel.of("web") == TaintLabel.of("web")
        assert TaintLabel.of("web") != TaintLabel.of("crm")

    def test_hashable_for_set_use(self) -> None:
        s = {TaintLabel.of("web"), TaintLabel.of("web"), TaintLabel.of("crm")}
        assert len(s) == 2

    def test_round_trip_dict(self) -> None:
        lbl = TaintLabel.of("web", "user_upload")
        assert TaintLabel.from_dict(lbl.to_dict()) == lbl

    def test_to_dict_sorted_for_stable_json(self) -> None:
        lbl = TaintLabel.of("z", "a", "m")
        assert lbl.to_dict() == {"sources": ["a", "m", "z"]}

    def test_from_dict_missing_sources_is_empty(self) -> None:
        assert TaintLabel.from_dict({}) == TaintLabel()


# --- ToolCall ------------------------------------------------------------------


class TestToolCall:
    def test_construct_minimal(self) -> None:
        c = ToolCall(tool_name="web_search")
        assert c.tool_name == "web_search"
        assert c.args == {}
        assert c.input_label == TaintLabel()
        assert c.agent_id is None
        assert c.call_id is None

    def test_construct_full(self) -> None:
        c = ToolCall(
            tool_name="send_email",
            args={"to": "x@y.com", "body": "hi"},
            input_label=TaintLabel.of("web"),
            agent_id="agent-1",
            call_id="call-42",
        )
        assert c.args == {"to": "x@y.com", "body": "hi"}
        assert c.input_label.subsumes(TaintLabel.of("web"))

    def test_value_equality(self) -> None:
        c1 = ToolCall("send_email", {"to": "x@y.com"})
        c2 = ToolCall("send_email", {"to": "x@y.com"})
        assert c1 == c2

    def test_inequality_on_taint(self) -> None:
        c1 = ToolCall("send_email", input_label=TaintLabel.of("web"))
        c2 = ToolCall("send_email", input_label=TaintLabel())
        assert c1 != c2

    def test_round_trip_dict(self) -> None:
        c = ToolCall(
            tool_name="web_search",
            args={"q": "test", "n": 5},
            input_label=TaintLabel.of("web"),
            agent_id="agent-1",
            call_id="call-42",
        )
        assert ToolCall.from_dict(c.to_dict()) == c

    def test_round_trip_dict_minimal(self) -> None:
        c = ToolCall(tool_name="noop")
        assert ToolCall.from_dict(c.to_dict()) == c

    def test_from_dict_handles_missing_input_label(self) -> None:
        d = {"tool_name": "noop"}
        c = ToolCall.from_dict(d)
        assert c.input_label == TaintLabel()


# --- Verdict + Decision --------------------------------------------------------


class TestVerdict:
    def test_string_values(self) -> None:
        assert Verdict.ALLOW.value == "allow"
        assert Verdict.DENY.value == "deny"
        assert Verdict.REVIEW.value == "review"

    def test_round_trip_via_string(self) -> None:
        assert Verdict("allow") is Verdict.ALLOW
        assert Verdict("deny") is Verdict.DENY


class TestDecision:
    def test_minimal_allow(self) -> None:
        d = Decision(verdict=Verdict.ALLOW)
        assert d.verdict is Verdict.ALLOW
        assert d.rule_id is None
        assert d.output_label == TaintLabel()

    def test_deny_with_reason(self) -> None:
        d = Decision(
            verdict=Verdict.DENY,
            rule_id="no-web-to-email",
            reason="web-tainted input to send_email",
        )
        assert d.verdict is Verdict.DENY
        assert d.rule_id == "no-web-to-email"
        assert "web" in d.reason

    def test_round_trip_dict(self) -> None:
        d = Decision(
            verdict=Verdict.REVIEW,
            rule_id="review-high-taint",
            reason="combined taint exceeds threshold",
            output_label=TaintLabel.of("web", "crm"),
        )
        assert Decision.from_dict(d.to_dict()) == d


# --- to_json / from_json -------------------------------------------------------


class TestJsonRoundTrip:
    def test_decision_round_trip(self) -> None:
        d = Decision(verdict=Verdict.DENY, reason="t", rule_id="r1")
        assert from_json(to_json(d), Decision) == d

    def test_tool_call_round_trip(self) -> None:
        c = ToolCall(tool_name="x", args={"a": 1}, input_label=TaintLabel.of("s"))
        assert from_json(to_json(c), ToolCall) == c

    def test_taint_label_round_trip(self) -> None:
        lbl = TaintLabel.of("a", "b")
        assert from_json(to_json(lbl), TaintLabel) == lbl

    def test_to_json_is_stable(self) -> None:
        # Same object → identical bytes (sorted keys).
        d = Decision(verdict=Verdict.ALLOW, output_label=TaintLabel.of("x", "y"))
        assert to_json(d) == to_json(d)
        # Output is parseable JSON.
        json.loads(to_json(d))

    def test_to_json_rejects_non_core_type(self) -> None:
        with pytest.raises(TypeError):
            to_json("plain string")  # type: ignore[arg-type]

    def test_from_json_rejects_non_core_type(self) -> None:
        with pytest.raises(TypeError):
            from_json("{}", str)
