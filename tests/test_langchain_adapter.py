"""Tests for the LangChain / LlamaIndex tool adapter (R21).

The adapter exposes :func:`wrap_langchain_tools`, a one-line helper that
mounts framework tool *objects* under a :class:`Gateway`. These tests
exercise it against hand-rolled fakes (neither ``langchain`` nor
``llama_index`` is a dependency of this project) covering allow / deny
paths, taint propagation, prefixing, resource-arg binding, the four
reserved kwargs, every supported invocation shape, the failure edges,
and the canonical web->email exfiltration scenario denied at send.
"""

from __future__ import annotations

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
    TaintCondition,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    wrap_langchain_tools,
)

# --------------------------------------------------------------------------- #
# Fixtures / fakes                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class _InvokeTool:
    """LangChain ``BaseTool``-style object: name + ``.invoke(dict)``."""

    name: str
    description: str = ""
    return_value: Any = None
    raises: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def invoke(self, arguments: dict[str, Any]) -> Any:
        self.calls.append(dict(arguments))
        if self.raises is not None:
            raise self.raises
        return self.return_value if self.return_value is not None else {
            "ok": True,
            "name": self.name,
        }


@dataclass
class _RunTool:
    """Legacy LangChain ``BaseTool``-style object: name + ``.run(dict)``."""

    name: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run(self, arguments: dict[str, Any]) -> Any:
        self.calls.append(dict(arguments))
        return {"ran": self.name}


@dataclass
class _Metadata:
    name: str


@dataclass
class _CallTool:
    """LlamaIndex ``BaseTool``-style object: ``.metadata.name`` + ``.call(**kw)``."""

    metadata: _Metadata
    calls: list[dict[str, Any]] = field(default_factory=list)

    def call(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return {"called": self.metadata.name}


@dataclass
class _FuncTool:
    """LangChain ``StructuredTool``-style object: name + ``.func(**kw)``."""

    name: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def func(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return {"func": self.name}


class _CallableTool:
    """Bare callable tool object: ``.name`` + ``__call__(**kw)``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return {"callable": self.name}


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
# Discovery & invocation shapes                                               #
# --------------------------------------------------------------------------- #


class TestDiscovery:
    def test_returns_one_callable_per_tool(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tools = wrap_langchain_tools(
            gw, [_InvokeTool(name="search"), _InvokeTool(name="read_file")]
        )
        assert set(tools) == {"search", "read_file"}
        assert all(callable(fn) for fn in tools.values())

    def test_invoke_shape_is_supported(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _InvokeTool(name="search")
        tools = wrap_langchain_tools(gw, [tool])
        result = tools["search"](query="apg", limit=5)
        assert result == {"ok": True, "name": "search"}
        assert tool.calls == [{"query": "apg", "limit": 5}]

    def test_run_shape_is_supported(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _RunTool(name="legacy")
        tools = wrap_langchain_tools(gw, [tool])
        assert tools["legacy"](q="x") == {"ran": "legacy"}
        assert tool.calls == [{"q": "x"}]

    def test_llamaindex_metadata_name_and_call_shape(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _CallTool(metadata=_Metadata(name="lookup"))
        tools = wrap_langchain_tools(gw, [tool])
        assert "lookup" in tools
        assert tools["lookup"](term="apg") == {"called": "lookup"}
        assert tool.calls == [{"term": "apg"}]

    def test_func_shape_is_supported(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _FuncTool(name="structured")
        tools = wrap_langchain_tools(gw, [tool])
        assert tools["structured"](a=1) == {"func": "structured"}
        assert tool.calls == [{"a": 1}]

    def test_bare_callable_shape_is_supported(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _CallableTool(name="plain")
        tools = wrap_langchain_tools(gw, [tool])
        assert tools["plain"](z=2) == {"callable": "plain"}
        assert tool.calls == [{"z": 2}]

    def test_prefix_namespaces_tools_without_changing_underlying_call(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _InvokeTool(name="read")
        tools = wrap_langchain_tools(gw, [tool], prefix="filesystem")
        assert "filesystem.read" in tools
        tools["filesystem.read"](path="/etc/passwd")
        assert tool.calls == [{"path": "/etc/passwd"}]


# --------------------------------------------------------------------------- #
# Argument forwarding & reserved kwargs                                       #
# --------------------------------------------------------------------------- #


class TestForwarding:
    def test_arguments_are_forwarded_verbatim_on_allow(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _InvokeTool(name="search")
        tools = wrap_langchain_tools(gw, [tool])
        tools["search"](query="apg", limit=5)
        assert tool.calls == [{"query": "apg", "limit": 5}]

    def test_reserved_kwargs_are_stripped_before_forwarding(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _InvokeTool(name="search")
        tools = wrap_langchain_tools(gw, [tool])
        tools["search"](
            query="x",
            apg_input_label=TaintLabel.of("user"),
            apg_agent_id="agent.research",
            apg_call_id="cid-1",
            apg_resource="https://example.com",
        )
        assert tool.calls == [{"query": "x"}]


# --------------------------------------------------------------------------- #
# Policy enforcement                                                          #
# --------------------------------------------------------------------------- #


class TestPolicy:
    def test_deny_raises_policy_denied_and_blocks_tool_call(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email")])
        send = _InvokeTool(name="send_email")
        search = _InvokeTool(name="search")
        tools = wrap_langchain_tools(gw, [send, search])

        with pytest.raises(PolicyDenied) as excinfo:
            tools["send_email"](to="x@y", body="b")
        assert excinfo.value.decision.verdict == Verdict.DENY
        assert excinfo.value.call.tool_name == "send_email"
        # No downstream call was made on deny.
        assert send.calls == []

        # Other tools still work.
        tools["search"](query="ok")
        assert search.calls == [{"query": "ok"}]

    def test_resource_args_binds_argument_for_selector_matching(self) -> None:
        policy = Policy(
            name="resource-test",
            rules=(
                Rule(
                    id="block-ops",
                    when=Selector(tool="send_email", resource="ops@*"),
                    effect=Effect(action=Action.DENY, reason="ops gated"),
                ),
                Rule(id="allow-rest", when=Selector(), effect=Effect(action=Action.ALLOW)),
            ),
        )
        gw = Gateway(policies=[policy])
        send = _InvokeTool(name="send_email")
        tools = wrap_langchain_tools(gw, [send], resource_args={"send_email": "to"})

        with pytest.raises(PolicyDenied):
            tools["send_email"](to="ops@example.com", body="b")
        tools["send_email"](to="alice@example.com", body="b")
        assert send.calls == [{"to": "alice@example.com", "body": "b"}]

    def test_apg_resource_override_takes_precedence_over_resource_args(self) -> None:
        policy = Policy(
            name="override-test",
            rules=(
                Rule(
                    id="deny-special",
                    when=Selector(tool="x", resource="forbidden"),
                    effect=Effect(action=Action.DENY, reason="nope"),
                ),
                Rule(id="allow-rest", when=Selector(), effect=Effect(action=Action.ALLOW)),
            ),
        )
        gw = Gateway(policies=[policy])
        tools = wrap_langchain_tools(
            gw, [_InvokeTool(name="x")], resource_args={"x": "a"}
        )
        with pytest.raises(PolicyDenied):
            tools["x"](a="ok", apg_resource="forbidden")


# --------------------------------------------------------------------------- #
# Taint propagation                                                           #
# --------------------------------------------------------------------------- #


class TestTaint:
    def test_taint_spec_is_registered_under_advertised_name(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrap_langchain_tools(
            gw,
            [_InvokeTool(name="search")],
            taint_specs={"search": ToolTaintSpec.of(adds=("web",))},
        )
        assert gw.tool_specs["search"].adds == TaintLabel.of("web")

    def test_taint_spec_is_registered_under_prefixed_name(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrap_langchain_tools(
            gw,
            [_InvokeTool(name="search")],
            taint_specs={"search": ToolTaintSpec.of(adds=("web",))},
            prefix="net",
        )
        assert "net.search" in gw.tool_specs
        assert gw.tool_specs["net.search"].adds == TaintLabel.of("web")
        assert "search" not in gw.tool_specs

    def test_propagation_adds_tool_sources_to_input_label(self) -> None:
        captured: list[Any] = []

        def writer(_call: ToolCall, decision: Any) -> None:
            captured.append(decision.output_label)

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        tools = wrap_langchain_tools(
            gw,
            [_InvokeTool(name="search")],
            taint_specs={"search": ToolTaintSpec.of(adds=("web",))},
        )
        tools["search"](query="x", apg_input_label=TaintLabel.of("user"))
        assert captured == [TaintLabel.of("user", "web")]


# --------------------------------------------------------------------------- #
# Audit-writer integration                                                    #
# --------------------------------------------------------------------------- #


class TestAudit:
    def test_audit_writer_sees_registered_name_and_arguments(self) -> None:
        captured: list[tuple[ToolCall, Any]] = []

        def writer(call: ToolCall, decision: Any) -> None:
            captured.append((call, decision))

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        tools = wrap_langchain_tools(gw, [_InvokeTool(name="read")], prefix="fs")
        tools["fs.read"](path="/tmp/x", apg_agent_id="alice")

        assert len(captured) == 1
        call, decision = captured[0]
        assert call.tool_name == "fs.read"
        assert call.args == {"path": "/tmp/x"}
        assert call.agent_id == "alice"
        assert decision.verdict == Verdict.ALLOW

    def test_audit_records_call_for_denied_call(self) -> None:
        captured: list[tuple[ToolCall, Any]] = []

        def writer(call: ToolCall, decision: Any) -> None:
            captured.append((call, decision))

        gw = Gateway(policies=[_deny_named("evil")], audit_writer=writer)
        evil = _InvokeTool(name="evil")
        tools = wrap_langchain_tools(gw, [evil])
        with pytest.raises(PolicyDenied):
            tools["evil"](kind="x")
        assert len(captured) == 1
        assert captured[0][1].verdict == Verdict.DENY
        assert evil.calls == []


# --------------------------------------------------------------------------- #
# Error edges                                                                 #
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_tool_without_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])

        class NoName:
            def invoke(self, arguments: dict[str, Any]) -> None:  # pragma: no cover
                return None

        with pytest.raises(ValueError, match="no 'name'"):
            wrap_langchain_tools(gw, [NoName()])

    def test_tool_with_empty_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        with pytest.raises(ValueError, match="non-empty string"):
            wrap_langchain_tools(gw, [_InvokeTool(name="")])

    def test_duplicate_names_raise_clearly(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        with pytest.raises(ValueError, match="duplicate"):
            wrap_langchain_tools(
                gw, [_InvokeTool(name="dup"), _InvokeTool(name="dup")]
            )

    def test_non_invocable_tool_raises_typeerror(self) -> None:
        gw = Gateway(policies=[_allow_all()])

        @dataclass
        class NotInvocable:
            name: str

        tools = wrap_langchain_tools(gw, [NotInvocable(name="inert")])
        with pytest.raises(TypeError, match="not invocable"):
            tools["inert"](a=1)

    def test_underlying_tool_exception_propagates_after_allow(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tool = _InvokeTool(name="boom", raises=RuntimeError("kapow"))
        tools = wrap_langchain_tools(gw, [tool])
        with pytest.raises(RuntimeError, match="kapow"):
            tools["boom"]()


# --------------------------------------------------------------------------- #
# Worked exfiltration scenario                                                #
# --------------------------------------------------------------------------- #


class TestExfiltrationScenario:
    def test_web_tainted_send_email_is_denied_at_send(self) -> None:
        # Policy: refuse send_email when input label includes "web".
        policy = Policy(
            name="no-web-exfil",
            rules=(
                Rule(
                    id="deny-web-send",
                    when=Selector(
                        tool="send_email",
                        taint=TaintCondition(any_of=("web",)),
                    ),
                    effect=Effect(action=Action.DENY, reason="web-tainted exfil"),
                ),
                Rule(id="allow-rest", when=Selector(), effect=Effect(action=Action.ALLOW)),
            ),
        )
        gw = Gateway(policies=[policy])
        search = _InvokeTool(name="search", return_value={"results": ["..."]})
        send_email = _InvokeTool(name="send_email")
        tools = wrap_langchain_tools(
            gw,
            [search, send_email],
            taint_specs={"search": ToolTaintSpec.of(adds=("web",))},
        )

        # 1) search() runs and its output carries 'web' taint.
        out = tools["search"](query="x", apg_input_label=TaintLabel.of("user"))
        assert out == {"results": ["..."]}

        # 2) send_email() under web-tainted input is denied at send.
        with pytest.raises(PolicyDenied) as excinfo:
            tools["send_email"](
                to="attacker@example.com",
                body="secret",
                apg_input_label=TaintLabel.of("web"),
            )
        assert excinfo.value.decision.rule_id == "deny-web-send"
        # The exfiltration send never reached the tool object.
        assert send_email.calls == []
