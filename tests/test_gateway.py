"""Tests for the reference monitor (R4): Gateway and wrap_tool decorator."""

from __future__ import annotations

import asyncio
import re

import pytest

from agent_policy_gateway import (
    Action,
    Decision,
    Effect,
    Gateway,
    Policy,
    PolicyDenied,
    PolicyReview,
    RateLimiter,
    Rule,
    Selector,
    TaintCondition,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    load_policy_str,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _list_audit() -> tuple[list[tuple[ToolCall, Decision]], object]:
    records: list[tuple[ToolCall, Decision]] = []

    def writer(call: ToolCall, decision: Decision) -> None:
        records.append((call, decision))

    return records, writer


def _allow_all(name: str = "allow-everything") -> Policy:
    return Policy(
        name=name,
        rules=(
            Rule(id="allow-all", when=Selector(), effect=Effect(action=Action.ALLOW)),
        ),
    )


def _deny_web_email() -> Policy:
    return Policy(
        name="deny-web-email",
        rules=(
            Rule(
                id="deny-web-to-email",
                when=Selector(
                    tool="send_email", taint=TaintCondition(any_of=("web",))
                ),
                effect=Effect(action=Action.DENY, reason="web-tainted email refused"),
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# Pure decision logic                                                         #
# --------------------------------------------------------------------------- #


class TestDecide:
    def test_default_allow_when_no_policies(self) -> None:
        gw = Gateway()
        d = gw.decide(ToolCall(tool_name="kb_lookup"))
        assert d.verdict == Verdict.ALLOW
        assert d.rule_id is None
        assert "default-allow" in d.reason

    def test_default_deny_when_configured_and_no_match(self) -> None:
        gw = Gateway(default_deny=True)
        d = gw.decide(ToolCall(tool_name="anything"))
        assert d.verdict == Verdict.DENY
        assert "default-deny" in d.reason

    def test_first_policy_wins_across_policies(self) -> None:
        # Policy A denies everything, Policy B allows. A is first => deny.
        deny_all = Policy(
            name="A",
            rules=(
                Rule(
                    id="deny",
                    when=Selector(),
                    effect=Effect(action=Action.DENY, reason="A"),
                ),
            ),
        )
        gw = Gateway(policies=[deny_all, _allow_all("B")])
        d = gw.decide(ToolCall(tool_name="x"))
        assert d.verdict == Verdict.DENY
        assert d.rule_id == "deny"
        assert d.reason == "A"

    def test_first_rule_wins_within_policy(self) -> None:
        p = Policy(
            name="ordered",
            rules=(
                Rule(
                    id="first",
                    when=Selector(tool="send_*"),
                    effect=Effect(action=Action.REVIEW, reason="first"),
                ),
                Rule(
                    id="second",
                    when=Selector(),
                    effect=Effect(action=Action.ALLOW),
                ),
            ),
        )
        gw = Gateway(policies=[p])
        d = gw.decide(ToolCall(tool_name="send_email"))
        assert d.rule_id == "first"
        assert d.verdict == Verdict.REVIEW

    def test_taint_propagation_through_spec(self) -> None:
        gw = Gateway()
        gw.register_tool(
            "redact",
            ToolTaintSpec.of(adds=("redactor",), declassifies=("pii",)),
        )
        call = ToolCall(tool_name="redact", input_label=TaintLabel.of("pii", "web"))
        d = gw.decide(call)
        assert d.output_label.sources == frozenset({"web", "redactor"})

    def test_taint_propagation_with_no_spec_is_pure_join(self) -> None:
        gw = Gateway()
        call = ToolCall(tool_name="echo", input_label=TaintLabel.of("a", "b"))
        d = gw.decide(call)
        assert d.output_label.sources == frozenset({"a", "b"})

    def test_rate_limit_is_allow_in_r4(self) -> None:
        p = Policy(
            name="rl",
            rules=(
                Rule(
                    id="rl",
                    when=Selector(tool="web_search"),
                    effect=Effect(action=Action.RATE_LIMIT, limit_per_minute=5),
                ),
            ),
        )
        gw = Gateway(policies=[p])
        d = gw.decide(ToolCall(tool_name="web_search"))
        assert d.verdict == Verdict.ALLOW
        assert d.rule_id == "rl"

    def test_resource_glob_matched(self) -> None:
        p = Policy(
            name="ext",
            rules=(
                Rule(
                    id="external",
                    when=Selector(tool="http_post", resource="https://*"),
                    effect=Effect(action=Action.REVIEW),
                ),
            ),
        )
        gw = Gateway(policies=[p])
        external = gw.decide(
            ToolCall(tool_name="http_post"), resource="https://api.example.com"
        )
        internal = gw.decide(
            ToolCall(tool_name="http_post"), resource="http://10.0.0.1/x"
        )
        assert external.verdict == Verdict.REVIEW
        assert internal.verdict == Verdict.ALLOW


# --------------------------------------------------------------------------- #
# execute() — running the underlying tool                                     #
# --------------------------------------------------------------------------- #


class TestExecute:
    def test_allow_calls_function(self) -> None:
        gw = Gateway()
        result, decision = gw.execute(
            ToolCall(tool_name="add"), lambda a, b: a + b, 2, 3
        )
        assert result == 5
        assert decision.verdict == Verdict.ALLOW

    def test_deny_raises_with_decision_attached(self) -> None:
        gw = Gateway(policies=[_deny_web_email()])
        call = ToolCall(
            tool_name="send_email", input_label=TaintLabel.of("web")
        )
        # Function should NOT be called.
        called: list[bool] = []

        def fn() -> str:
            called.append(True)
            return "sent"

        with pytest.raises(PolicyDenied) as ei:
            gw.execute(call, fn)
        assert called == []
        assert ei.value.decision.verdict == Verdict.DENY
        assert ei.value.decision.rule_id == "deny-web-to-email"
        assert ei.value.call.tool_name == "send_email"
        assert "web-tainted" in str(ei.value)

    def test_review_raises_policy_review(self) -> None:
        p = Policy(
            name="review",
            rules=(
                Rule(
                    id="r",
                    when=Selector(),
                    effect=Effect(action=Action.REVIEW, reason="needs human"),
                ),
            ),
        )
        gw = Gateway(policies=[p])
        with pytest.raises(PolicyReview) as ei:
            gw.execute(ToolCall(tool_name="x"), lambda: None)
        assert ei.value.decision.verdict == Verdict.REVIEW
        assert "needs human" in str(ei.value)

    def test_audit_writer_is_called_on_allow_and_deny(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(policies=[_deny_web_email()], audit_writer=writer)

        # ALLOW path
        gw.execute(ToolCall(tool_name="kb_lookup"), lambda: "ok")
        # DENY path
        with pytest.raises(PolicyDenied):
            gw.execute(
                ToolCall(
                    tool_name="send_email", input_label=TaintLabel.of("web")
                ),
                lambda: None,
            )
        assert len(records) == 2
        assert records[0][1].verdict == Verdict.ALLOW
        assert records[1][1].verdict == Verdict.DENY

    def test_audit_writer_failure_aborts_call(self) -> None:
        def boom(call: ToolCall, decision: Decision) -> None:
            raise RuntimeError("disk full")

        called: list[bool] = []
        gw = Gateway(audit_writer=boom)
        with pytest.raises(RuntimeError, match="disk full"):
            gw.execute(
                ToolCall(tool_name="x"),
                lambda: called.append(True) or "ok",
            )
        assert called == []  # tool never ran


# --------------------------------------------------------------------------- #
# wrap_tool — decorator behaviour                                             #
# --------------------------------------------------------------------------- #


class TestWrapTool:
    def test_bare_decorator_uses_function_name(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(audit_writer=writer)

        @gw.wrap_tool
        def ping(host: str) -> str:
            return f"pong:{host}"

        assert ping("localhost") == "pong:localhost"
        assert records[0][0].tool_name == "ping"
        assert records[0][0].args == {"host": "localhost"}

    def test_explicit_tool_name_overrides_function_name(self) -> None:
        gw = Gateway()
        records, writer = _list_audit()
        gw.audit_writer = writer

        @gw.wrap_tool(tool_name="external_send")
        def _internal(to: str) -> str:
            return to

        _internal("a@b")
        assert records[0][0].tool_name == "external_send"

    def test_reserved_kwargs_are_stripped_before_fn(self) -> None:
        seen_kwargs: dict[str, object] = {}
        gw = Gateway()

        @gw.wrap_tool
        def echo(**kw: object) -> dict[str, object]:
            seen_kwargs.update(kw)
            return kw

        result = echo(
            x=1,
            apg_input_label=TaintLabel.of("web"),
            apg_agent_id="agent.research",
            apg_call_id="abc",
            apg_resource="https://x",
        )
        assert "apg_input_label" not in result
        assert "apg_agent_id" not in seen_kwargs
        assert "apg_call_id" not in seen_kwargs
        assert "apg_resource" not in seen_kwargs
        assert result == {"x": 1}

    def test_input_label_and_agent_id_propagate_into_call(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(audit_writer=writer)

        @gw.wrap_tool
        def fetch(url: str) -> str:
            return f"<body of {url}>"

        fetch(
            "https://x",
            apg_input_label=TaintLabel.of("user_input"),
            apg_agent_id="agent.research",
        )
        call = records[0][0]
        assert call.tool_name == "fetch"
        assert call.input_label == TaintLabel.of("user_input")
        assert call.agent_id == "agent.research"
        # auto call_id present (uuid4 hex == 32 chars)
        assert call.call_id is not None
        assert re.fullmatch(r"[0-9a-f]{32}", call.call_id)

    def test_explicit_call_id_is_preserved(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(audit_writer=writer)

        @gw.wrap_tool
        def t() -> int:
            return 1

        t(apg_call_id="trace-42")
        assert records[0][0].call_id == "trace-42"

    def test_taint_spec_registered_via_decorator(self) -> None:
        gw = Gateway()

        @gw.wrap_tool(tool_name="web_search", taint_spec=ToolTaintSpec.of(adds=("web",)))
        def web_search(q: str) -> str:
            return f"results for {q}"

        records, writer = _list_audit()
        gw.audit_writer = writer
        web_search("kittens")
        out_label = records[0][1].output_label
        assert "web" in out_label.sources

    def test_resource_arg_picks_up_value_from_kwargs(self) -> None:
        p = Policy(
            name="ext-only",
            rules=(
                Rule(
                    id="ext",
                    when=Selector(tool="http_post", resource="https://*"),
                    effect=Effect(action=Action.DENY, reason="external"),
                ),
            ),
        )
        gw = Gateway(policies=[p])

        @gw.wrap_tool(tool_name="http_post", resource_arg="url")
        def http_post(url: str, body: str) -> str:
            return body

        # internal: allowed
        assert http_post("http://localhost/", "ok") == "ok"
        # external: denied
        with pytest.raises(PolicyDenied):
            http_post("https://api.example.com/", "ok")

    def test_resource_arg_works_for_positional_call(self) -> None:
        p = Policy(
            name="ext-only",
            rules=(
                Rule(
                    id="ext",
                    when=Selector(tool="http_post", resource="https://*"),
                    effect=Effect(action=Action.DENY, reason="external"),
                ),
            ),
        )
        gw = Gateway(policies=[p])

        @gw.wrap_tool(tool_name="http_post", resource_arg="url")
        def http_post(url: str, body: str) -> str:
            return body

        with pytest.raises(PolicyDenied):
            http_post("https://api.example.com/", "ok")

    def test_apg_resource_overrides_resource_arg(self) -> None:
        p = Policy(
            name="https-only",
            rules=(
                Rule(
                    id="ext",
                    when=Selector(tool="http", resource="https://*"),
                    effect=Effect(action=Action.DENY, reason="external"),
                ),
            ),
        )
        gw = Gateway(policies=[p])

        @gw.wrap_tool(tool_name="http", resource_arg="url")
        def http(url: str) -> str:
            return url

        # The url arg is internal but the explicit resource is external.
        with pytest.raises(PolicyDenied):
            http("http://internal/", apg_resource="https://x.example")

    def test_default_input_label_is_empty(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(audit_writer=writer)

        @gw.wrap_tool
        def t() -> str:
            return "ok"

        t()
        assert records[0][0].input_label == TaintLabel()

    def test_wrap_tool_can_wrap_existing_function_directly(self) -> None:
        gw = Gateway()

        def add(a: int, b: int) -> int:
            return a + b

        wrapped = gw.wrap_tool(add)
        assert wrapped(1, 2) == 3

    def test_wrap_tool_preserves_metadata(self) -> None:
        gw = Gateway()

        @gw.wrap_tool
        def documented(x: int) -> int:
            """Returns x."""
            return x

        assert documented.__name__ == "documented"
        assert (documented.__doc__ or "").strip() == "Returns x."


# --------------------------------------------------------------------------- #
# End-to-end: indirect-prompt-injection exfiltration is denied                #
# --------------------------------------------------------------------------- #


class TestExfiltrationScenario:
    POLICY_YAML = """
version: 1
name: exfil
rules:
  - id: deny-web-to-email
    when:
      tool: send_email
      taint:
        any_of: [web]
    effect:
      action: deny
      reason: Web-tainted email blocked.
  - id: allow-internal
    when:
      tool: kb_lookup
    effect:
      action: allow
"""

    def _make_gateway(self) -> tuple[Gateway, list[tuple[ToolCall, Decision]]]:
        records, writer = _list_audit()
        gw = Gateway(
            policies=[load_policy_str(self.POLICY_YAML)],
            audit_writer=writer,
        )
        gw.register_tool("web_search", ToolTaintSpec.of(adds=("web",)))
        gw.register_tool("summarize", ToolTaintSpec.of())
        gw.register_tool("send_email", ToolTaintSpec.of())
        return gw, records

    def test_web_to_email_chain_is_denied_at_send(self) -> None:
        gw, records = self._make_gateway()

        @gw.wrap_tool(tool_name="web_search", taint_spec=ToolTaintSpec.of(adds=("web",)))
        def web_search(q: str) -> str:
            return "results"

        @gw.wrap_tool(tool_name="summarize")
        def summarize(text: str) -> str:
            return text[:80]

        @gw.wrap_tool(tool_name="send_email")
        def send_email(to: str, body: str) -> str:
            return "sent"

        # Step 1: web_search adds 'web' taint.
        ws_records_before = len(records)
        web_search("anything")
        assert records[ws_records_before][1].verdict == Verdict.ALLOW
        assert records[ws_records_before][1].output_label.sources == frozenset({"web"})

        # Step 2: summarize propagates the taint forward.
        summarize_label = records[ws_records_before][1].output_label
        summarize("body", apg_input_label=summarize_label)
        assert records[-1][1].output_label.sources == frozenset({"web"})

        # Step 3: send_email is refused because taint includes 'web'.
        with pytest.raises(PolicyDenied) as ei:
            send_email(
                "ops@example.com",
                "exfiltrated body",
                apg_input_label=records[-1][1].output_label,
            )
        assert ei.value.decision.rule_id == "deny-web-to-email"

        # And the audit log captured the refusal too.
        assert records[-1][1].verdict == Verdict.DENY
        assert records[-1][0].tool_name == "send_email"

    def test_internal_lookup_unaffected(self) -> None:
        gw, _records = self._make_gateway()

        @gw.wrap_tool(tool_name="kb_lookup")
        def kb_lookup(q: str) -> str:
            return "internal answer"

        assert kb_lookup("anything") == "internal answer"


# --------------------------------------------------------------------------- #
# R16: rate-limit enforcement through the stateful execute path               #
# --------------------------------------------------------------------------- #


class _FakeClock:
    """Manually advanced clock for deterministic window tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _rate_limited_gateway(limit: int, clock: _FakeClock):
    policy = Policy(
        name="rl",
        rules=(
            Rule(
                id="throttle",
                when=Selector(tool="web_search"),
                effect=Effect(action=Action.RATE_LIMIT, limit_per_minute=limit),
            ),
        ),
    )
    records, writer = _list_audit()
    gw = Gateway(
        policies=[policy],
        audit_writer=writer,
        rate_limiter=RateLimiter(window_seconds=60, clock=clock),
    )
    return gw, records


class TestRateLimitEnforcement:
    def test_allows_n_then_denies_n_plus_one(self) -> None:
        clock = _FakeClock()
        gw, records = _rate_limited_gateway(3, clock)

        def fn() -> str:
            return "ok"

        # First 3 calls within the window succeed.
        for _ in range(3):
            value, decision = gw.execute(ToolCall(tool_name="web_search"), fn)
            assert value == "ok"
            assert decision.verdict == Verdict.ALLOW
            assert decision.rule_id == "throttle"

        # The 4th call inside the same window is refused.
        with pytest.raises(PolicyDenied) as exc:
            gw.execute(ToolCall(tool_name="web_search"), fn)
        assert exc.value.decision.verdict == Verdict.DENY
        assert exc.value.decision.rule_id == "throttle"
        assert "rate limit exceeded" in exc.value.decision.reason

    def test_refusal_is_audited(self) -> None:
        clock = _FakeClock()
        gw, records = _rate_limited_gateway(1, clock)
        fn = lambda: "ok"  # noqa: E731
        gw.execute(ToolCall(tool_name="web_search"), fn)
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="web_search"), fn)
        # Both the allow and the deny are recorded; the last record is the refusal.
        assert len(records) == 2
        assert records[-1][1].verdict == Verdict.DENY

    def test_denied_call_does_not_run_the_tool(self) -> None:
        clock = _FakeClock()
        gw, _ = _rate_limited_gateway(1, clock)
        calls: list[int] = []

        def fn() -> str:
            calls.append(1)
            return "ran"

        gw.execute(ToolCall(tool_name="web_search"), fn)
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="web_search"), fn)
        assert calls == [1]  # the second (denied) call never invoked fn

    def test_window_expiry_allows_again(self) -> None:
        clock = _FakeClock()
        gw, _ = _rate_limited_gateway(2, clock)
        fn = lambda: "ok"  # noqa: E731
        gw.execute(ToolCall(tool_name="web_search"), fn)
        gw.execute(ToolCall(tool_name="web_search"), fn)
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="web_search"), fn)
        # Past the 60s window the oldest slots expire and calls flow again.
        clock.advance(60.1)
        value, decision = gw.execute(ToolCall(tool_name="web_search"), fn)
        assert value == "ok"
        assert decision.verdict == Verdict.ALLOW

    def test_limit_is_per_agent_tool_key(self) -> None:
        clock = _FakeClock()
        gw, _ = _rate_limited_gateway(1, clock)
        fn = lambda: "ok"  # noqa: E731
        # agent.a exhausts its budget.
        gw.execute(ToolCall(tool_name="web_search", agent_id="agent.a"), fn)
        with pytest.raises(PolicyDenied):
            gw.execute(ToolCall(tool_name="web_search", agent_id="agent.a"), fn)
        # agent.b has an independent budget under the same rule.
        value, decision = gw.execute(
            ToolCall(tool_name="web_search", agent_id="agent.b"), fn
        )
        assert decision.verdict == Verdict.ALLOW

    def test_decide_peek_does_not_consume(self) -> None:
        clock = _FakeClock()
        gw, _ = _rate_limited_gateway(1, clock)
        call = ToolCall(tool_name="web_search")
        # Pure decide() can be called repeatedly without burning the budget.
        for _ in range(5):
            assert gw.decide(call).verdict == Verdict.ALLOW
        fn = lambda: "ok"  # noqa: E731
        value, decision = gw.execute(call, fn)
        assert decision.verdict == Verdict.ALLOW
        # Now the single slot is consumed; decide reflects the full window.
        assert gw.decide(call).verdict == Verdict.DENY

    def test_async_execute_enforces_limit(self) -> None:
        clock = _FakeClock()
        gw, _ = _rate_limited_gateway(2, clock)

        async def fn() -> str:
            return "ok"

        async def scenario() -> None:
            await gw.aexecute(ToolCall(tool_name="web_search"), fn)
            await gw.aexecute(ToolCall(tool_name="web_search"), fn)
            with pytest.raises(PolicyDenied):
                await gw.aexecute(ToolCall(tool_name="web_search"), fn)

        asyncio.run(scenario())
