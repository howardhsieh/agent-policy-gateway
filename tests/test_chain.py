"""Tests for chain-level policies (R53).

Covers the ``chain:`` selector condition — ``PriorCallMatcher`` /
``ProvenanceMatcher`` / ``ProvenanceCondition`` / ``ChainCondition`` and
their YAML schema — the gateway's ``track_history`` session recording,
the untracked-history fail-safe (chain rules referencing prior calls
never match without a history), the ``WatchedPolicy`` duck-type, the
CLI (`explain --prior`, lint W002-for-chains, conservative W001), the
AgentDojo adapter's per-episode history reset, and the shipped
``policies/agentdojo-chain.yaml`` example.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from agent_policy_gateway.cli import main
from agent_policy_gateway.core import (
    CallHistoryEntry,
    Provenance,
    ProvenanceEntry,
    TaintLabel,
    ToolCall,
    Verdict,
)
from agent_policy_gateway.gateway import Gateway, PolicyDenied
from agent_policy_gateway.policy import (
    ChainCondition,
    PolicyError,
    PriorCallMatcher,
    ProvenanceCondition,
    ProvenanceMatcher,
    Selector,
    load_policy,
    load_policy_str,
)
from agent_policy_gateway.reload import watch_policy
from agent_policy_gateway.taint import ToolTaintSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES_DIR = REPO_ROOT / "policies"


def _entry(
    tool: str = "web_fetch",
    verdict: Verdict = Verdict.ALLOW,
    *sources: str,
    resource: str | None = None,
) -> CallHistoryEntry:
    return CallHistoryEntry(
        tool_name=tool,
        verdict=verdict,
        output_label=TaintLabel.of(*sources),
        resource=resource,
    )


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# CallHistoryEntry                                                            #
# --------------------------------------------------------------------------- #


class TestCallHistoryEntry:
    def test_round_trips_through_dict(self) -> None:
        entry = CallHistoryEntry(
            tool_name="send_email",
            verdict=Verdict.DENY,
            output_label=TaintLabel.of("web"),
            agent_id="agent.a",
            call_id="c1",
            resource="mail://ops",
        )
        assert CallHistoryEntry.from_dict(entry.to_dict()) == entry

    def test_defaults(self) -> None:
        entry = CallHistoryEntry(tool_name="t", verdict=Verdict.ALLOW)
        assert entry.output_label == TaintLabel()
        assert entry.agent_id is None
        assert entry.call_id is None
        assert entry.resource is None


# --------------------------------------------------------------------------- #
# PriorCallMatcher                                                            #
# --------------------------------------------------------------------------- #


class TestPriorCallMatcher:
    def test_empty_matcher_matches_any_entry(self) -> None:
        assert PriorCallMatcher().matches_entry(_entry())

    def test_tool_glob(self) -> None:
        m = PriorCallMatcher(tool="get_*")
        assert m.matches_entry(_entry("get_webpage"))
        assert not m.matches_entry(_entry("send_email"))

    def test_verdict_exact(self) -> None:
        m = PriorCallMatcher(verdict=Verdict.ALLOW)
        assert m.matches_entry(_entry(verdict=Verdict.ALLOW))
        assert not m.matches_entry(_entry(verdict=Verdict.DENY))

    def test_unset_verdict_matches_denied_attempts_too(self) -> None:
        assert PriorCallMatcher(tool="web_*").matches_entry(
            _entry("web_fetch", Verdict.DENY)
        )

    def test_source_globs_the_output_label(self) -> None:
        m = PriorCallMatcher(source="crm.*")
        assert m.matches_entry(_entry("t", Verdict.ALLOW, "crm.contact"))
        assert not m.matches_entry(_entry("t", Verdict.ALLOW, "web"))
        assert not m.matches_entry(_entry("t"))  # empty label

    def test_source_sees_every_dimension(self) -> None:
        entry = CallHistoryEntry(
            tool_name="t",
            verdict=Verdict.ALLOW,
            output_label=TaintLabel.of_dimensions(integrity=["web"]),
        )
        assert PriorCallMatcher(source="web").matches_entry(entry)

    def test_resource_glob_needs_a_recorded_resource(self) -> None:
        m = PriorCallMatcher(resource="https://*")
        assert m.matches_entry(_entry(resource="https://x.example"))
        assert not m.matches_entry(_entry(resource="ftp://x"))
        assert not m.matches_entry(_entry())  # no resource recorded

    def test_fields_are_conjunctive(self) -> None:
        m = PriorCallMatcher(tool="get_*", source="web", verdict=Verdict.ALLOW)
        assert m.matches_entry(_entry("get_webpage", Verdict.ALLOW, "web"))
        assert not m.matches_entry(_entry("get_webpage", Verdict.DENY, "web"))
        assert not m.matches_entry(_entry("get_webpage", Verdict.ALLOW, "pii"))


# --------------------------------------------------------------------------- #
# Provenance matching                                                         #
# --------------------------------------------------------------------------- #


class TestProvenanceMatcher:
    def test_empty_matcher_matches_any_entry(self) -> None:
        assert ProvenanceMatcher().matches_entry(
            ProvenanceEntry(source="web", tool_name="web_fetch")
        )

    def test_source_and_tool_globs(self) -> None:
        entry = ProvenanceEntry(source="rss.feed", tool_name="fetch_rss")
        assert ProvenanceMatcher(source="rss.*").matches_entry(entry)
        assert ProvenanceMatcher(tool="fetch_*").matches_entry(entry)
        assert not ProvenanceMatcher(source="web").matches_entry(entry)
        assert not ProvenanceMatcher(
            source="rss.*", tool="send_*"
        ).matches_entry(entry)


class TestProvenanceCondition:
    def test_empty_condition_matches_everything(self) -> None:
        cond = ProvenanceCondition()
        assert cond.is_empty()
        assert cond.matches(Provenance())

    def test_any_of_needs_a_matching_entry(self) -> None:
        cond = ProvenanceCondition(any_of=(ProvenanceMatcher(source="web"),))
        chain = Provenance((ProvenanceEntry(source="web", tool_name="f"),))
        assert cond.matches(chain)
        assert not cond.matches(Provenance())
        assert not cond.matches(
            Provenance((ProvenanceEntry(source="pii", tool_name="f"),))
        )

    def test_none_of_forbids_matching_entries(self) -> None:
        cond = ProvenanceCondition(none_of=(ProvenanceMatcher(tool="web_*"),))
        assert cond.matches(Provenance())
        assert not cond.matches(
            Provenance((ProvenanceEntry(source="web", tool_name="web_fetch"),))
        )


# --------------------------------------------------------------------------- #
# ChainCondition                                                              #
# --------------------------------------------------------------------------- #


class TestChainCondition:
    def test_empty_is_empty_and_matches(self) -> None:
        cond = ChainCondition()
        assert cond.is_empty()
        assert not cond.needs_history()
        assert cond.matches(ToolCall(tool_name="t"), history=None)

    def test_history_clauses_need_a_history(self) -> None:
        call = ToolCall(tool_name="t")
        any_prior = ChainCondition(any_prior=(PriorCallMatcher(),))
        no_prior = ChainCondition(no_prior=(PriorCallMatcher(tool="x"),))
        assert any_prior.needs_history() and no_prior.needs_history()
        # None = untracked: neither clause can be verified, so no match —
        # even no_prior, which an *empty* history would satisfy.
        assert not any_prior.matches(call, history=None)
        assert not no_prior.matches(call, history=None)
        assert not any_prior.matches(call, history=[])
        assert no_prior.matches(call, history=[])

    def test_any_prior_matches_any_entry_against_any_matcher(self) -> None:
        cond = ChainCondition(
            any_prior=(
                PriorCallMatcher(tool="get_*"),
                PriorCallMatcher(source="crm.*"),
            )
        )
        call = ToolCall(tool_name="send_email")
        assert cond.matches(call, history=[_entry("get_webpage")])
        assert cond.matches(call, history=[_entry("t", Verdict.ALLOW, "crm.x")])
        assert not cond.matches(call, history=[_entry("read_file")])

    def test_no_prior_forbids_matching_entries(self) -> None:
        cond = ChainCondition(no_prior=(PriorCallMatcher(tool="web_*"),))
        call = ToolCall(tool_name="send_email")
        assert cond.matches(call, history=[_entry("read_file")])
        assert not cond.matches(call, history=[_entry("web_fetch")])

    def test_provenance_only_condition_ignores_history(self) -> None:
        cond = ChainCondition(
            provenance=ProvenanceCondition(
                any_of=(ProvenanceMatcher(source="web"),)
            )
        )
        assert not cond.needs_history()
        call = ToolCall(
            tool_name="t",
            input_provenance=Provenance(
                (ProvenanceEntry(source="web", tool_name="f"),)
            ),
        )
        # Works with history=None: it constrains the call's input, not
        # the session history.
        assert cond.matches(call, history=None)
        assert not cond.matches(ToolCall(tool_name="t"), history=None)

    def test_clauses_are_conjunctive(self) -> None:
        cond = ChainCondition(
            any_prior=(PriorCallMatcher(tool="get_*"),),
            no_prior=(PriorCallMatcher(tool="sanitize"),),
        )
        call = ToolCall(tool_name="t")
        assert cond.matches(call, history=[_entry("get_webpage")])
        assert not cond.matches(
            call, history=[_entry("get_webpage"), _entry("sanitize")]
        )


# --------------------------------------------------------------------------- #
# Selector + schema                                                           #
# --------------------------------------------------------------------------- #


class TestSelectorChain:
    def test_selector_threads_history(self) -> None:
        sel = Selector(chain=ChainCondition(any_prior=(PriorCallMatcher(),)))
        call = ToolCall(tool_name="t")
        assert sel.matches(call, history=[_entry()])
        assert not sel.matches(call, history=[])
        assert not sel.matches(call)  # untracked default

    def test_chain_is_conjunctive_with_other_clauses(self) -> None:
        sel = Selector(
            tool="send_*",
            chain=ChainCondition(any_prior=(PriorCallMatcher(tool="get_*"),)),
        )
        history = [_entry("get_webpage")]
        assert sel.matches(ToolCall(tool_name="send_email"), history=history)
        assert not sel.matches(ToolCall(tool_name="read_file"), history=history)


class TestChainSchema:
    YAML = """
version: 1
name: chain-demo
rules:
  - id: deny-send-after-web
    when:
      tool: send_email
      chain:
        any_prior:
          - tool: "get_*"
            source: web
            verdict: allow
            resource: "https://*"
        no_prior:
          - tool: sanitize
        provenance:
          any_of:
            - {source: web, tool: "get_*"}
          none_of: []
    effect:
      action: deny
      reason: chain rule
"""

    def test_full_chain_condition_loads(self) -> None:
        policy = load_policy_str(self.YAML)
        chain = policy.rules[0].when.chain
        assert chain is not None
        assert chain.any_prior[0].tool == "get_*"
        assert chain.any_prior[0].verdict is Verdict.ALLOW
        assert chain.no_prior[0].tool == "sanitize"
        assert chain.provenance is not None
        assert chain.provenance.any_of[0].source == "web"

    def test_chain_defaults_to_none(self) -> None:
        policy = load_policy_str(
            "version: 1\nname: p\nrules:\n"
            "  - id: r\n    effect: {action: allow}\n"
        )
        assert policy.rules[0].when.chain is None

    def test_unknown_chain_field_rejected(self) -> None:
        with pytest.raises(PolicyError):
            load_policy_str(
                "version: 1\nname: p\nrules:\n"
                "  - id: r\n"
                "    when: {chain: {any_call: []}}\n"
                "    effect: {action: deny}\n"
            )

    def test_unknown_matcher_field_rejected(self) -> None:
        with pytest.raises(PolicyError):
            load_policy_str(
                "version: 1\nname: p\nrules:\n"
                "  - id: r\n"
                "    when: {chain: {any_prior: [{tool_name: x}]}}\n"
                "    effect: {action: deny}\n"
            )

    def test_bad_verdict_rejected(self) -> None:
        with pytest.raises(PolicyError):
            load_policy_str(
                "version: 1\nname: p\nrules:\n"
                "  - id: r\n"
                "    when: {chain: {any_prior: [{verdict: blocked}]}}\n"
                "    effect: {action: deny}\n"
            )

    def test_empty_matcher_is_valid_any_call(self) -> None:
        policy = load_policy_str(
            "version: 1\nname: p\nrules:\n"
            "  - id: r\n"
            "    when: {chain: {any_prior: [{}]}}\n"
            "    effect: {action: deny}\n"
        )
        chain = policy.rules[0].when.chain
        assert chain is not None
        assert chain.matches(ToolCall(tool_name="t"), history=[_entry()])


# --------------------------------------------------------------------------- #
# Gateway history                                                             #
# --------------------------------------------------------------------------- #


def _chain_policy() -> str:
    return """
version: 1
name: chain-gw
rules:
  - id: deny-send-after-web
    when:
      tool: send_email
      chain:
        any_prior:
          - source: web
            verdict: allow
    effect:
      action: deny
      reason: a web-sourced call precedes this send
"""


class TestGatewayHistory:
    def test_track_history_defaults_off_and_chain_rules_inert(self) -> None:
        gw = Gateway(policies=[load_policy_str(_chain_policy())])
        assert gw.track_history is False
        gw.execute(ToolCall(tool_name="web_search"), lambda: "ok")
        # Even a matching sequence cannot fire the chain rule: no history.
        result, decision = gw.execute(ToolCall(tool_name="send_email"), lambda: "sent")
        assert decision.verdict is Verdict.ALLOW
        assert gw.call_history() == ()

    def test_execute_records_history(self) -> None:
        gw = Gateway(track_history=True)
        gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))
        gw.execute(
            ToolCall(tool_name="web_search", call_id="c1", agent_id="a"),
            lambda: "ok",
            resource="https://x",
        )
        (entry,) = gw.call_history("a")
        assert entry.tool_name == "web_search"
        assert entry.verdict is Verdict.ALLOW
        assert entry.output_label == TaintLabel.of("web")
        assert entry.call_id == "c1"
        assert entry.agent_id == "a"
        assert entry.resource == "https://x"

    def test_denied_attempt_is_recorded(self) -> None:
        policy = load_policy_str(
            "version: 1\nname: p\nrules:\n"
            "  - id: no-send\n"
            "    when: {tool: send_email}\n"
            "    effect: {action: deny}\n"
        )
        gw = Gateway(policies=[policy], track_history=True)
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="send_email"), lambda: "x")
        (entry,) = gw.call_history()
        assert entry.verdict is Verdict.DENY

    def test_decide_is_pure_and_records_nothing(self) -> None:
        gw = Gateway(track_history=True)
        gw.decide(ToolCall(tool_name="t"))
        assert gw.call_history() == ()

    def test_reset_history_clears_every_agent(self) -> None:
        gw = Gateway(track_history=True)
        gw.execute(ToolCall(tool_name="t", agent_id="a"), lambda: 1)
        gw.execute(ToolCall(tool_name="t", agent_id="b"), lambda: 1)
        gw.reset_history()
        assert gw.call_history("a") == ()
        assert gw.call_history("b") == ()

    def test_history_is_keyed_by_agent_id(self) -> None:
        gw = Gateway(policies=[load_policy_str(_chain_policy())], track_history=True)
        gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))
        gw.execute(ToolCall(tool_name="web_search", agent_id="a"), lambda: "ok")
        # Agent b never made a web call: its send_email is allowed.
        _, decision = gw.execute(
            ToolCall(tool_name="send_email", agent_id="b"), lambda: "sent"
        )
        assert decision.verdict is Verdict.ALLOW
        # Agent a's is denied by the chain rule.
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="send_email", agent_id="a"), lambda: "x")

    def test_chain_rule_end_to_end(self) -> None:
        gw = Gateway(policies=[load_policy_str(_chain_policy())], track_history=True)
        gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))
        # Before any web call: allowed (a call never sees itself in history).
        _, decision = gw.execute(ToolCall(tool_name="send_email"), lambda: "sent")
        assert decision.verdict is Verdict.ALLOW
        gw.execute(ToolCall(tool_name="web_search"), lambda: "page")
        with pytest.raises(PolicyDenied) as exc_info:
            gw.execute(ToolCall(tool_name="send_email"), lambda: "x")
        assert exc_info.value.decision.rule_id == "deny-send-after-web"

    def test_denied_web_call_does_not_fire_allow_scoped_matcher(self) -> None:
        deny_web = (
            "  - id: no-web\n"
            "    when: {tool: web_search}\n"
            "    effect: {action: deny}\n"
        )
        policy = load_policy_str(_chain_policy() + deny_web)
        gw = Gateway(policies=[policy], track_history=True)
        gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="web_search"), lambda: "x")
        # The denied attempt is in history (its would-be output label
        # included), but the matcher requires verdict *allow* — a denial
        # never executed, so it must not arm the chain rule.
        _, decision = gw.execute(ToolCall(tool_name="send_email"), lambda: "sent")
        assert decision.verdict is Verdict.ALLOW

    def test_aexecute_records_history(self) -> None:
        import asyncio

        gw = Gateway(policies=[load_policy_str(_chain_policy())], track_history=True)
        gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))

        async def tool() -> str:
            return "ok"

        async def scenario() -> None:
            await gw.aexecute(ToolCall(tool_name="web_search"), tool)
            assert len(gw.call_history()) == 1
            with pytest.raises(PolicyDenied):
                await gw.aexecute(ToolCall(tool_name="send_email"), tool)

        asyncio.run(scenario())

    def test_redacted_call_recorded_with_redact_verdict(self) -> None:
        policy = load_policy_str(
            """
version: 1
name: p
rules:
  - id: mask
    when: {tool: send_email}
    effect:
      action: redact
      redact:
        fields: [body]
"""
        )
        gw = Gateway(policies=[policy], track_history=True)
        gw.execute(
            ToolCall(tool_name="send_email", args={"body": "hi"}),
            lambda body: body,
            body="hi",
        )
        (entry,) = gw.call_history()
        assert entry.verdict is Verdict.REDACT

    def test_provenance_rule_matches_input_chain(self) -> None:
        policy = load_policy_str(
            """
version: 1
name: p
rules:
  - id: deny-web-derived
    when:
      tool: send_email
      chain:
        provenance:
          any_of:
            - {tool: "web_*"}
    effect: {action: deny}
"""
        )
        gw = Gateway(policies=[policy])
        prov = Provenance((ProvenanceEntry(source="web", tool_name="web_fetch"),))
        with pytest.raises(PolicyDenied):
            gw.execute(
                ToolCall(tool_name="send_email", input_provenance=prov),
                lambda: "x",
            )
        _, decision = gw.execute(ToolCall(tool_name="send_email"), lambda: "sent")
        assert decision.verdict is Verdict.ALLOW


class TestWatchedPolicyChain:
    def test_first_match_accepts_history(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text(_chain_policy(), encoding="utf-8")
        watched = watch_policy(path)
        call = ToolCall(tool_name="send_email")
        history = [_entry("web_search", Verdict.ALLOW, "web")]
        rule = watched.first_match(call, history=history)
        assert rule is not None and rule.id == "deny-send-after-web"
        assert watched.first_match(call, history=[]) is None
        assert watched.first_match(call) is None

    def test_gateway_with_watched_chain_policy(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text(_chain_policy(), encoding="utf-8")
        gw = Gateway(policies=[watch_policy(path)], track_history=True)
        gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))
        gw.execute(ToolCall(tool_name="web_search"), lambda: "ok")
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="send_email"), lambda: "x")


# --------------------------------------------------------------------------- #
# CLI: lint + explain --prior                                                 #
# --------------------------------------------------------------------------- #


class TestChainLint:
    def _lint(self, tmp_path: Path, body: str) -> tuple[int, str]:
        path = tmp_path / "p.yaml"
        path.write_text(body, encoding="utf-8")
        rc, out, _ = _run(["policy", "lint", str(path)])
        return rc, out

    def test_any_prior_forbidden_by_no_prior_is_w002(self, tmp_path: Path) -> None:
        rc, out = self._lint(
            tmp_path,
            """
version: 1
name: p
rules:
  - id: impossible
    when:
      chain:
        any_prior: [{tool: web_fetch}]
        no_prior: [{tool: web_fetch}]
    effect: {action: deny}
""",
        )
        assert rc == 3
        assert "W002" in out and "impossible" in out

    def test_provenance_any_of_forbidden_is_w002(self, tmp_path: Path) -> None:
        rc, out = self._lint(
            tmp_path,
            """
version: 1
name: p
rules:
  - id: impossible
    when:
      chain:
        provenance:
          any_of: [{source: web}]
          none_of: [{source: web}]
    effect: {action: deny}
""",
        )
        assert rc == 3
        assert "W002" in out

    def test_satisfiable_chain_is_clean(self, tmp_path: Path) -> None:
        rc, out = self._lint(
            tmp_path,
            """
version: 1
name: p
rules:
  - id: ok
    when:
      chain:
        any_prior: [{tool: web_fetch}]
        no_prior: [{tool: sanitize}]
    effect: {action: deny}
""",
        )
        assert rc == 0
        assert "OK" in out

    def test_earlier_chain_rule_never_claims_generality(self, tmp_path: Path) -> None:
        # Without the R53 conservatism the earlier rule (same tool, plus a
        # chain constraint) would be mistaken for at-least-as-general.
        rc, out = self._lint(
            tmp_path,
            """
version: 1
name: p
rules:
  - id: chained
    when:
      tool: send_email
      chain:
        any_prior: [{source: web}]
    effect: {action: deny}
  - id: plain
    when:
      tool: send_email
    effect: {action: review}
""",
        )
        assert rc == 0

    def test_later_chain_rule_shadowed_by_plain_earlier_rule(
        self, tmp_path: Path
    ) -> None:
        rc, out = self._lint(
            tmp_path,
            """
version: 1
name: p
rules:
  - id: plain
    when:
      tool: send_email
    effect: {action: deny}
  - id: chained
    when:
      tool: send_email
      chain:
        any_prior: [{source: web}]
    effect: {action: review}
""",
        )
        assert rc == 3
        assert "W001" in out and "chained" in out


CHAIN_POLICY_YAML = """
version: 1
name: explain-chain
rules:
  - id: deny-send-after-web
    when:
      tool: send_email
      chain:
        any_prior:
          - source: web
            verdict: allow
        no_prior:
          - tool: sanitize
    effect:
      action: deny
      reason: web before send
"""


class TestExplainPrior:
    @pytest.fixture()
    def policy_file(self, tmp_path: Path) -> str:
        path = tmp_path / "p.yaml"
        path.write_text(CHAIN_POLICY_YAML, encoding="utf-8")
        return str(path)

    def test_prior_satisfies_chain_rule(self, policy_file: str) -> None:
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                policy_file,
                "--tool",
                "send_email",
                "--prior",
                "web_fetch,source=web",
            ]
        )
        assert rc == 0
        assert "[MATCH ] deny-send-after-web" in out
        assert "prior:  web_fetch(allow)" in out

    def test_no_prior_flag_means_empty_history(self, policy_file: str) -> None:
        rc, out, _ = _run(
            ["policy", "explain", policy_file, "--tool", "send_email"]
        )
        assert rc == 0
        assert "no rule matched" in out
        assert "no prior call (of 0 recorded)" in out

    def test_no_prior_clause_offender_is_named(self, policy_file: str) -> None:
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                policy_file,
                "--tool",
                "send_email",
                "--prior",
                "web_fetch,source=web",
                "--prior",
                "sanitize",
            ]
        )
        assert rc == 0
        assert "'sanitize'" in out and "no_prior" in out

    def test_prior_verdict_deny_does_not_satisfy_allow_matcher(
        self, policy_file: str
    ) -> None:
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                policy_file,
                "--tool",
                "send_email",
                "--prior",
                "web_fetch,source=web,verdict=deny",
            ]
        )
        assert rc == 0
        assert "no rule matched" in out

    def test_prior_resource_and_dimension_sources_parse(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text(
            """
version: 1
name: p
rules:
  - id: r
    when:
      chain:
        any_prior:
          - resource: "https://*"
            source: web
    effect: {action: deny}
""",
            encoding="utf-8",
        )
        rc, out, _ = _run(
            [
                "policy",
                "explain",
                str(path),
                "--tool",
                "t",
                "--prior",
                "fetch,source=integ:web,resource=https://x.example",
            ]
        )
        assert rc == 0
        assert "[MATCH ] r" in out

    @pytest.mark.parametrize(
        "bad",
        [
            "tool=oops",  # first item must be a bare tool name
            "fetch,verdict=blocked",  # unknown verdict
            "fetch,color=red",  # unknown key
            "fetch,source",  # missing =
        ],
    )
    def test_malformed_prior_is_usage_error(self, policy_file: str, bad: str) -> None:
        with pytest.raises(SystemExit):
            _run(
                [
                    "policy",
                    "explain",
                    policy_file,
                    "--tool",
                    "t",
                    "--prior",
                    bad,
                ]
            )


# --------------------------------------------------------------------------- #
# AgentDojo adapter + shipped chain policy                                    #
# --------------------------------------------------------------------------- #


class _FakeRuntime:
    def __init__(self) -> None:
        self.functions: dict[str, object] = {}
        self.calls: list[str] = []

    def run_function(
        self, env: object, function: str, kwargs: dict, raise_on_error: bool = False
    ) -> tuple[object, str | None]:
        self.calls.append(function)
        return f"{function}-result", None


class TestAgentDojoHistory:
    def test_reset_taint_also_resets_tracked_history(self) -> None:
        from agent_policy_gateway.agentdojo_adapter import wrap_agentdojo_runtime

        gw = Gateway(track_history=True)
        gated = wrap_agentdojo_runtime(gw, _FakeRuntime())
        gated.run_function(None, "read_inbox", {})
        assert len(gw.call_history("agentdojo")) == 1
        gated.reset_taint()
        assert gw.call_history("agentdojo") == ()

    def test_reset_taint_leaves_untracked_gateway_alone(self) -> None:
        from agent_policy_gateway.agentdojo_adapter import wrap_agentdojo_runtime

        gw = Gateway()
        gated = wrap_agentdojo_runtime(gw, _FakeRuntime())
        gated.run_function(None, "read_inbox", {})
        gated.reset_taint()  # must not raise
        assert gated.taint_label == TaintLabel()

    def test_chain_policy_closes_reader_borne_exfiltration(self) -> None:
        """The policies/agentdojo-chain.yaml scenario on a fake runtime."""
        from agent_policy_gateway.agentdojo_adapter import wrap_agentdojo_runtime

        gw = Gateway(
            policies=[load_policy(POLICIES_DIR / "agentdojo-chain.yaml")],
            track_history=True,
        )
        runtime = _FakeRuntime()
        gated = wrap_agentdojo_runtime(
            gw,
            runtime,
            taint_specs={
                "read_channel_messages": ToolTaintSpec.of(
                    adds=("agentdojo:untrusted",)
                )
            },
        )
        # First fetch of an episode: no untrusted read yet -> allowed.
        _, error = gated.run_function(None, "get_webpage", {"url": "https://ok"})
        assert error is None
        gated.reset_taint()
        # After an untrusted read, the fetch is refused by the chain rule.
        _, error = gated.run_function(None, "read_channel_messages", {})
        assert error is None
        _, error = gated.run_function(None, "get_webpage", {"url": "https://evil"})
        assert error is not None
        assert "deny-web-fetch-after-untrusted-read" in error
        assert runtime.calls.count("get_webpage") == 1


class TestChainPolicyFile:
    def test_loads_and_extends_the_baseline(self) -> None:
        chain = load_policy(POLICIES_DIR / "agentdojo-chain.yaml")
        base = load_policy(POLICIES_DIR / "agentdojo.yaml")
        assert chain.name == "agentdojo-chain"
        # Lockstep: the baseline's rules, byte-for-byte, then the chain rules.
        assert chain.rules[: len(base.rules)] == base.rules
        extra = [r.id for r in chain.rules[len(base.rules) :]]
        assert extra == ["deny-web-fetch-after-untrusted-read"]

    def test_chain_rule_requires_an_executed_untrusted_call(self) -> None:
        policy = load_policy(POLICIES_DIR / "agentdojo-chain.yaml")
        call = ToolCall(tool_name="get_webpage")
        tainted = [_entry("read_inbox", Verdict.ALLOW, "agentdojo:untrusted")]
        rule = policy.first_match(call, history=tainted)
        assert rule is not None
        assert rule.id == "deny-web-fetch-after-untrusted-read"
        assert policy.first_match(call, history=[]) is None
        denied = [_entry("read_inbox", Verdict.DENY, "agentdojo:untrusted")]
        assert policy.first_match(call, history=denied) is None
