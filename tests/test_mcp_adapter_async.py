"""Tests for the async MCP adapter (R9).

Mirrors :mod:`tests.test_mcp_adapter` against an ``async def`` session
that matches the shape of the real ``mcp.ClientSession``: both
``list_tools`` and ``call_tool`` are coroutines. The synchronous
adapter helpers (``_iter_tool_descriptors``, ``_tool_name``) are
shared between adapters, so most error-edge coverage carries over —
this file focuses on the async-specific paths.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_policy_gateway import (
    Action,
    Effect,
    Gateway,
    Policy,
    PolicyDenied,
    Rule,
    Selector,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    wrap_mcp_session_async,
)

# --------------------------------------------------------------------------- #
# Fixtures / fakes                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class _Descriptor:
    name: str


@dataclass
class _ListToolsResult:
    """Mimics ``mcp.types.ListToolsResult`` (a ``.tools``-bearing wrapper)."""

    tools: list[Any]


@dataclass
class _AsyncFakeSession:
    """Async session mirroring the real ``mcp.ClientSession`` shape."""

    descriptors: list[Any]
    return_values: dict[str, Any] = field(default_factory=dict)
    raise_on: dict[str, Exception] = field(default_factory=dict)
    use_wrapper: bool = False
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def list_tools(self) -> Any:
        await asyncio.sleep(0)
        if self.use_wrapper:
            return _ListToolsResult(tools=list(self.descriptors))
        return list(self.descriptors)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        await asyncio.sleep(0)
        self.calls.append((name, dict(arguments)))
        if name in self.raise_on:
            raise self.raise_on[name]
        return self.return_values.get(name, {"ok": True, "name": name})


def _allow_all() -> Policy:
    return Policy(
        name="allow-all",
        rules=(
            Rule(id="allow-all", when=Selector(), effect=Effect(action=Action.ALLOW)),
        ),
    )


def _deny_named(tool_name: str) -> Policy:
    return Policy(
        name="deny-one",
        rules=(
            Rule(
                id="deny-by-name",
                when=Selector(tool=tool_name),
                effect=Effect(action=Action.DENY, reason="forbidden"),
            ),
            Rule(id="allow-rest", when=Selector(), effect=Effect(action=Action.ALLOW)),
        ),
    )


# --------------------------------------------------------------------------- #
# Discovery                                                                   #
# --------------------------------------------------------------------------- #


class TestDiscovery:
    def test_returns_one_callable_per_advertised_tool(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(
            descriptors=[_Descriptor(name="search"), _Descriptor(name="read_file")]
        )
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        assert set(tools) == {"search", "read_file"}
        assert all(asyncio.iscoroutinefunction(fn) for fn in tools.values())

    def test_dict_descriptors_are_supported(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(
            descriptors=[{"name": "echo", "description": "ignored"}]
        )
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        assert "echo" in tools
        asyncio.run(tools["echo"](message="hi"))
        assert session.calls == [("echo", {"message": "hi"})]

    def test_list_tools_with_tools_attribute_is_unpacked(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(
            descriptors=[_Descriptor(name="alpha"), _Descriptor(name="beta")],
            use_wrapper=True,
        )
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        assert set(tools) == {"alpha", "beta"}

    def test_prefix_namespaces_tools_without_changing_advertised_calls(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="read")])
        tools = asyncio.run(
            wrap_mcp_session_async(gw, session, prefix="filesystem")
        )
        assert "filesystem.read" in tools
        asyncio.run(tools["filesystem.read"](path="/etc/passwd"))
        # The MCP server still receives the *advertised* name.
        assert session.calls == [("read", {"path": "/etc/passwd"})]


# --------------------------------------------------------------------------- #
# Argument forwarding & reserved kwargs                                       #
# --------------------------------------------------------------------------- #


class TestForwarding:
    def test_arguments_are_forwarded_verbatim_on_allow(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="search")])
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        result = asyncio.run(tools["search"](query="apg", limit=5))
        assert result == {"ok": True, "name": "search"}
        assert session.calls == [("search", {"query": "apg", "limit": 5})]

    def test_reserved_kwargs_are_stripped_before_forwarding(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="search")])
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        asyncio.run(
            tools["search"](
                query="x",
                apg_input_label=TaintLabel.of("user"),
                apg_agent_id="agent.research",
                apg_call_id="cid-1",
                apg_resource="https://example.com",
            )
        )
        name, args = session.calls[0]
        assert name == "search"
        assert args == {"query": "x"}


# --------------------------------------------------------------------------- #
# Policy enforcement                                                          #
# --------------------------------------------------------------------------- #


class TestPolicy:
    def test_deny_raises_policy_denied_and_blocks_session_call(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email")])
        session = _AsyncFakeSession(
            descriptors=[
                _Descriptor(name="send_email"),
                _Descriptor(name="search"),
            ]
        )
        tools = asyncio.run(wrap_mcp_session_async(gw, session))

        with pytest.raises(PolicyDenied) as excinfo:
            asyncio.run(tools["send_email"](to="x@y", body="b"))
        assert excinfo.value.decision.verdict == Verdict.DENY
        assert excinfo.value.call.tool_name == "send_email"
        # No downstream call was made on deny.
        assert session.calls == []

        # Other tools still work.
        asyncio.run(tools["search"](query="ok"))
        assert session.calls == [("search", {"query": "ok"})]

    def test_resource_args_binds_argument_for_selector_matching(self) -> None:
        policy = Policy(
            name="resource-test",
            rules=(
                Rule(
                    id="block-ops",
                    when=Selector(tool="send_email", resource="ops@*"),
                    effect=Effect(action=Action.DENY, reason="ops gated"),
                ),
                Rule(
                    id="allow-rest",
                    when=Selector(),
                    effect=Effect(action=Action.ALLOW),
                ),
            ),
        )
        gw = Gateway(policies=[policy])
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="send_email")])
        tools = asyncio.run(
            wrap_mcp_session_async(
                gw, session, resource_args={"send_email": "to"}
            )
        )

        with pytest.raises(PolicyDenied):
            asyncio.run(tools["send_email"](to="ops@example.com", body="b"))
        # A different recipient is allowed.
        asyncio.run(tools["send_email"](to="alice@example.com", body="b"))
        assert session.calls == [
            ("send_email", {"to": "alice@example.com", "body": "b"})
        ]

    def test_apg_resource_override_takes_precedence_over_resource_args(self) -> None:
        policy = Policy(
            name="override-test",
            rules=(
                Rule(
                    id="deny-special",
                    when=Selector(tool="x", resource="forbidden"),
                    effect=Effect(action=Action.DENY, reason="nope"),
                ),
                Rule(
                    id="allow-rest",
                    when=Selector(),
                    effect=Effect(action=Action.ALLOW),
                ),
            ),
        )
        gw = Gateway(policies=[policy])
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="x")])
        tools = asyncio.run(
            wrap_mcp_session_async(gw, session, resource_args={"x": "a"})
        )
        with pytest.raises(PolicyDenied):
            asyncio.run(tools["x"](a="ok", apg_resource="forbidden"))


# --------------------------------------------------------------------------- #
# Taint propagation                                                           #
# --------------------------------------------------------------------------- #


class TestTaint:
    def test_taint_spec_is_registered_under_advertised_name(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="search")])
        asyncio.run(
            wrap_mcp_session_async(
                gw,
                session,
                taint_specs={"search": ToolTaintSpec.of(adds=("web",))},
            )
        )
        assert gw.tool_specs["search"].adds == TaintLabel.of("web")

    def test_taint_spec_is_registered_under_prefixed_name(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="search")])
        asyncio.run(
            wrap_mcp_session_async(
                gw,
                session,
                taint_specs={"search": ToolTaintSpec.of(adds=("web",))},
                prefix="net",
            )
        )
        assert "net.search" in gw.tool_specs
        assert gw.tool_specs["net.search"].adds == TaintLabel.of("web")
        assert "search" not in gw.tool_specs

    def test_propagation_adds_tool_sources_to_input_label(self) -> None:
        captured: list[Any] = []

        def writer(_call: ToolCall, decision: Any) -> None:
            captured.append(decision.output_label)

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="search")])
        tools = asyncio.run(
            wrap_mcp_session_async(
                gw, session, taint_specs={"search": ToolTaintSpec.of(adds=("web",))}
            )
        )
        asyncio.run(
            tools["search"](query="x", apg_input_label=TaintLabel.of("user"))
        )
        assert captured == [TaintLabel.of("user", "web")]


# --------------------------------------------------------------------------- #
# Audit-writer integration                                                    #
# --------------------------------------------------------------------------- #


class TestAudit:
    def test_sync_audit_writer_sees_registered_name_and_arguments(self) -> None:
        captured: list[tuple[ToolCall, Any]] = []

        def writer(call: ToolCall, decision: Any) -> None:
            captured.append((call, decision))

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="read")])
        tools = asyncio.run(wrap_mcp_session_async(gw, session, prefix="fs"))
        asyncio.run(tools["fs.read"](path="/tmp/x", apg_agent_id="alice"))

        assert len(captured) == 1
        call, decision = captured[0]
        assert call.tool_name == "fs.read"
        assert call.args == {"path": "/tmp/x"}
        assert call.agent_id == "alice"
        assert decision.verdict == Verdict.ALLOW

    def test_async_audit_writer_is_awaited_per_call(self) -> None:
        captured: list[tuple[ToolCall, Any]] = []

        async def writer(call: ToolCall, decision: Any) -> None:
            await asyncio.sleep(0)
            captured.append((call, decision))

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="ping")])
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        asyncio.run(tools["ping"](msg="hi"))
        assert len(captured) == 1
        assert captured[0][0].tool_name == "ping"
        assert captured[0][1].verdict == Verdict.ALLOW

    def test_audit_records_call_for_denied_call(self) -> None:
        captured: list[tuple[ToolCall, Any]] = []

        def writer(call: ToolCall, decision: Any) -> None:
            captured.append((call, decision))

        gw = Gateway(policies=[_deny_named("evil")], audit_writer=writer)
        session = _AsyncFakeSession(descriptors=[_Descriptor(name="evil")])
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        with pytest.raises(PolicyDenied):
            asyncio.run(tools["evil"](kind="x"))
        assert len(captured) == 1
        assert captured[0][1].verdict == Verdict.DENY
        # Underlying session was not touched.
        assert session.calls == []


# --------------------------------------------------------------------------- #
# Error edges                                                                 #
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_invalid_descriptor_missing_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        bad = _AsyncFakeSession(descriptors=[{"description": "no name here"}])
        with pytest.raises(ValueError, match="missing 'name'"):
            asyncio.run(wrap_mcp_session_async(gw, bad))

    def test_invalid_descriptor_with_empty_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        bad = _AsyncFakeSession(descriptors=[_Descriptor(name="")])
        with pytest.raises(ValueError, match="non-empty string"):
            asyncio.run(wrap_mcp_session_async(gw, bad))

    def test_invalid_descriptor_without_name_attribute_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        bad = _AsyncFakeSession(descriptors=[object()])
        with pytest.raises(ValueError, match="no 'name' attribute"):
            asyncio.run(wrap_mcp_session_async(gw, bad))

    def test_session_missing_list_tools_raises_typeerror(self) -> None:
        gw = Gateway(policies=[_allow_all()])

        class NoListTools:
            async def call_tool(self, name: str, arguments: dict[str, Any]) -> None:
                return None  # pragma: no cover

        with pytest.raises(TypeError, match="list_tools"):
            asyncio.run(wrap_mcp_session_async(gw, NoListTools()))

    def test_session_missing_call_tool_raises_typeerror(self) -> None:
        gw = Gateway(policies=[_allow_all()])

        class NoCallTool:
            async def list_tools(self) -> list[Any]:
                return []

        with pytest.raises(TypeError, match="call_tool"):
            asyncio.run(wrap_mcp_session_async(gw, NoCallTool()))

    def test_duplicate_advertised_names_raise_clearly(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(
            descriptors=[_Descriptor(name="dup"), _Descriptor(name="dup")]
        )
        with pytest.raises(ValueError, match="duplicate"):
            asyncio.run(wrap_mcp_session_async(gw, session))

    def test_underlying_session_exception_propagates_after_allow(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        session = _AsyncFakeSession(
            descriptors=[_Descriptor(name="boom")],
            raise_on={"boom": RuntimeError("kapow")},
        )
        tools = asyncio.run(wrap_mcp_session_async(gw, session))
        with pytest.raises(RuntimeError, match="kapow"):
            asyncio.run(tools["boom"]())
