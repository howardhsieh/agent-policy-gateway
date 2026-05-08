"""Tests for the taint propagation algebra (R2).

Covers:
* Lattice properties of `join`: identity (with empty), commutativity,
  associativity, idempotence; the empty-args case yielding bottom.
* Lattice order properties of `subsumes`: reflexive, antisymmetric,
  transitive.
* `flows_to` semantics for sink decisions.
* `propagate` rules: empty inputs + tool sources, multi-input join,
  declassification stripping known sources, declassification with a
  source that wasn't present being a no-op.
* End-to-end worked example: a tool chain
  `web_search → summarize → send_email` is denied because the email
  sink does not allow `web` taint. The `summarize` tool is a transparent
  propagator (no spec); the `web_search` tool adds `{"web"}`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_policy_gateway import (
    Decision,
    TaintLabel,
    ToolTaintSpec,
    Verdict,
    flows_to,
    join,
    join_all,
    propagate,
    subsumes,
)

# --- join: lattice algebra ----------------------------------------------------


class TestJoin:
    def test_no_args_is_bottom(self) -> None:
        assert join() == TaintLabel()
        assert join().is_empty()

    def test_single_arg_is_identity(self) -> None:
        a = TaintLabel.of("web")
        assert join(a) == a

    def test_join_with_empty_is_identity(self) -> None:
        a = TaintLabel.of("web")
        assert join(a, TaintLabel()) == a
        assert join(TaintLabel(), a) == a

    def test_commutative(self) -> None:
        a = TaintLabel.of("web")
        b = TaintLabel.of("crm")
        assert join(a, b) == join(b, a)

    def test_associative(self) -> None:
        a = TaintLabel.of("a")
        b = TaintLabel.of("b")
        c = TaintLabel.of("c")
        assert join(join(a, b), c) == join(a, join(b, c))

    def test_idempotent(self) -> None:
        a = TaintLabel.of("web")
        assert join(a, a) == a
        assert join(a, a, a) == a

    def test_n_ary(self) -> None:
        out = join(
            TaintLabel.of("web"),
            TaintLabel.of("crm"),
            TaintLabel.of("user_upload"),
        )
        assert out.sources == {"web", "crm", "user_upload"}

    def test_join_all_equivalent_to_varargs(self) -> None:
        labels = [TaintLabel.of("a"), TaintLabel.of("b"), TaintLabel.of("c")]
        assert join_all(labels) == join(*labels)

    def test_join_all_empty_iterable_is_bottom(self) -> None:
        assert join_all([]) == TaintLabel()


# --- subsumes / flows_to: order ----------------------------------------------


class TestSubsumes:
    def test_reflexive(self) -> None:
        a = TaintLabel.of("web")
        assert subsumes(a, a)

    def test_antisymmetric(self) -> None:
        a = TaintLabel.of("web")
        b = TaintLabel.of("web")
        assert subsumes(a, b) and subsumes(b, a)
        assert a == b

    def test_transitive(self) -> None:
        a = TaintLabel.of("web", "crm", "fs")
        b = TaintLabel.of("web", "crm")
        c = TaintLabel.of("web")
        assert subsumes(a, b) and subsumes(b, c)
        assert subsumes(a, c)

    def test_strict(self) -> None:
        a = TaintLabel.of("web", "crm")
        b = TaintLabel.of("web")
        assert subsumes(a, b)
        assert not subsumes(b, a)

    def test_bottom_is_least(self) -> None:
        bottom = TaintLabel()
        a = TaintLabel.of("web")
        assert subsumes(a, bottom)
        assert subsumes(bottom, bottom)
        assert not subsumes(bottom, a)


class TestFlowsTo:
    def test_empty_flows_anywhere(self) -> None:
        assert flows_to(TaintLabel(), TaintLabel())
        assert flows_to(TaintLabel(), TaintLabel.of("web"))

    def test_tainted_flows_into_matching_sink(self) -> None:
        assert flows_to(TaintLabel.of("web"), TaintLabel.of("web", "crm"))

    def test_tainted_blocked_at_strict_sink(self) -> None:
        assert not flows_to(TaintLabel.of("web"), TaintLabel())
        assert not flows_to(TaintLabel.of("web", "crm"), TaintLabel.of("web"))


# --- propagate: per-call rule -------------------------------------------------


class TestPropagate:
    def test_no_inputs_no_spec_is_bottom(self) -> None:
        assert propagate([]) == TaintLabel()

    def test_no_spec_is_pure_join(self) -> None:
        out = propagate([TaintLabel.of("web"), TaintLabel.of("crm")])
        assert out.sources == {"web", "crm"}

    def test_spec_adds_tool_sources(self) -> None:
        spec = ToolTaintSpec.of(adds=["web"])
        out = propagate([], spec)
        assert out == TaintLabel.of("web")

    def test_spec_adds_join_with_inputs(self) -> None:
        spec = ToolTaintSpec.of(adds=["fs"])
        out = propagate([TaintLabel.of("web"), TaintLabel.of("crm")], spec)
        assert out.sources == {"web", "crm", "fs"}

    def test_declassify_strips_known_source(self) -> None:
        spec = ToolTaintSpec.of(declassifies=["web"])
        out = propagate([TaintLabel.of("web", "crm")], spec)
        assert out == TaintLabel.of("crm")

    def test_declassify_unknown_source_is_noop(self) -> None:
        spec = ToolTaintSpec.of(declassifies=["pii"])
        out = propagate([TaintLabel.of("web")], spec)
        assert out == TaintLabel.of("web")

    def test_declassify_after_adds(self) -> None:
        # A tool that *both* adds and declassifies: declassification wins on
        # the overlap (it operates on the post-add set).
        spec = ToolTaintSpec.of(adds=["web"], declassifies=["web"])
        out = propagate([TaintLabel.of("crm")], spec)
        assert out == TaintLabel.of("crm")

    def test_propagate_is_pure_returns_new_label(self) -> None:
        a = TaintLabel.of("web")
        out = propagate([a])
        # Inputs unchanged (frozen dataclasses, but verify identity model).
        assert a == TaintLabel.of("web")
        assert out == a


# --- ToolTaintSpec ------------------------------------------------------------


class TestToolTaintSpec:
    def test_default_is_transparent(self) -> None:
        spec = ToolTaintSpec()
        assert spec.adds == TaintLabel()
        assert spec.declassifies == TaintLabel()

    def test_of_constructs_from_strings(self) -> None:
        spec = ToolTaintSpec.of(adds=["web"], declassifies=["pii"])
        assert spec.adds == TaintLabel.of("web")
        assert spec.declassifies == TaintLabel.of("pii")

    def test_value_equality(self) -> None:
        a = ToolTaintSpec.of(adds=["web"])
        b = ToolTaintSpec.of(adds=["web"])
        assert a == b


# --- Worked end-to-end exfiltration example ----------------------------------


@dataclass(frozen=True)
class _ToolNode:
    """Tiny helper standing in for the (R4) Gateway, just for this test."""

    name: str
    spec: ToolTaintSpec


def _run_chain(chain: list[_ToolNode]) -> TaintLabel:
    """Simulate a left-to-right tool chain. Each tool receives the previous
    output's label as its single input label."""
    current = TaintLabel()
    for node in chain:
        current = propagate([current], node.spec)
    return current


class TestWorkedExfiltrationExample:
    """`web_search → summarize → send_email` must be refused.

    `web_search` produces `web`-tainted output. `summarize` is a transparent
    propagator (no spec), so the taint survives. `send_email` is a sink that
    only admits the empty label — so the policy decision is DENY with a
    reason that names the offending source.
    """

    def test_chain_carries_web_taint_to_email_sink(self) -> None:
        web_search = _ToolNode("web_search", ToolTaintSpec.of(adds=["web"]))
        summarize = _ToolNode("summarize", ToolTaintSpec())
        chain_output = _run_chain([web_search, summarize])
        assert chain_output == TaintLabel.of("web")

    def test_email_sink_refuses_web_tainted_input(self) -> None:
        web_search = _ToolNode("web_search", ToolTaintSpec.of(adds=["web"]))
        summarize = _ToolNode("summarize", ToolTaintSpec())
        chain_output = _run_chain([web_search, summarize])

        # send_email's policy: only the empty label may flow in.
        email_allowed = TaintLabel()

        if flows_to(chain_output, email_allowed):
            decision = Decision(verdict=Verdict.ALLOW, output_label=chain_output)
        else:
            offending = sorted(chain_output.sources - email_allowed.sources)
            decision = Decision(
                verdict=Verdict.DENY,
                rule_id="no-web-to-email",
                reason=f"input carries disallowed sources {offending} for sink send_email",
                output_label=chain_output,
            )

        assert decision.verdict is Verdict.DENY
        assert decision.rule_id == "no-web-to-email"
        assert "web" in decision.reason

    def test_declassifier_unblocks_email_sink(self) -> None:
        # A trusted summarizer that *declassifies* `web` would let the chain
        # proceed. This documents the intended use of declassification.
        web_search = _ToolNode("web_search", ToolTaintSpec.of(adds=["web"]))
        trusted_summary = _ToolNode(
            "trusted_summary",
            ToolTaintSpec.of(declassifies=["web"]),
        )
        chain_output = _run_chain([web_search, trusted_summary])
        assert chain_output.is_empty()
        assert flows_to(chain_output, TaintLabel())  # empty sink is fine
