"""Tests for the OpenAI function-calling adapter (R7).

The adapter exposes :func:`openai_tool_specs` (build the JSON tool descriptors
sent in the API request) and :func:`wrap_openai_tools` (mount Python tools
under a :class:`Gateway` and return a name -> callable map), with
:func:`dispatch_openai_tool_call` translating model-produced tool_calls into
gateway-mediated invocations and structured tool-role responses. These tests
exercise the adapter end-to-end against ad-hoc fakes — the real ``openai``
SDK is intentionally *not* a dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from agent_policy_gateway import (
    Action,
    Effect,
    Gateway,
    OpenAITool,
    OpenAIToolCallError,
    Policy,
    Rule,
    Selector,
    TaintCondition,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    dispatch_openai_tool_call,
    dispatch_openai_tool_calls,
    openai_tool_specs,
    wrap_openai_tools,
)

# --------------------------------------------------------------------------- #
# Helpers / fakes                                                             #
# --------------------------------------------------------------------------- #


def _allow_all() -> Policy:
    return Policy(
        name="allow-all",
        rules=(
            Rule(id="allow", when=Selector(), effect=Effect(action=Action.ALLOW)),
        ),
    )


def _deny_named(tool_name: str, *, reason: str = "forbidden") -> Policy:
    return Policy(
        name="deny-one",
        rules=(
            Rule(
                id="deny-by-name",
                when=Selector(tool=tool_name),
                effect=Effect(action=Action.DENY, reason=reason),
            ),
            Rule(id="allow-rest", when=Selector(), effect=Effect(action=Action.ALLOW)),
        ),
    )


def _review_named(tool_name: str) -> Policy:
    return Policy(
        name="review-one",
        rules=(
            Rule(
                id="review-by-name",
                when=Selector(tool=tool_name),
                effect=Effect(action=Action.REVIEW, reason="needs human"),
            ),
            Rule(id="allow-rest", when=Selector(), effect=Effect(action=Action.ALLOW)),
        ),
    )


def _make_tool_call(
    *,
    name: str,
    arguments: dict[str, Any] | str | None = None,
    call_id: str = "call_1",
) -> dict[str, Any]:
    if arguments is None:
        arg_str = "{}"
    elif isinstance(arguments, str):
        arg_str = arguments
    else:
        arg_str = json.dumps(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arg_str},
    }


@dataclass
class _AttrFunction:
    name: str
    arguments: str


@dataclass
class _AttrToolCall:
    """Mimics the openai-python SDK's pydantic-style tool_call object."""

    id: str
    function: _AttrFunction
    type: str = "function"


# Sample Python tools used across the test cases.

_SEARCH_PARAMS = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["query"],
}

_SEND_PARAMS = {
    "type": "object",
    "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
    "required": ["to", "body"],
}


def _search_fn(query: str, limit: int = 5) -> dict[str, Any]:
    return {"query": query, "results": [f"r{i}" for i in range(limit)]}


def _send_fn(to: str, body: str) -> dict[str, Any]:
    return {"sent_to": to, "size": len(body)}


# --------------------------------------------------------------------------- #
# OpenAITool construction                                                     #
# --------------------------------------------------------------------------- #


class TestOpenAIToolConstruction:
    def test_minimal_construction(self) -> None:
        t = OpenAITool(
            name="search",
            description="d",
            parameters=_SEARCH_PARAMS,
            function=_search_fn,
        )
        assert t.name == "search"
        assert t.resource_arg is None
        assert t.taint_spec is None

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            OpenAITool(
                name="", description="d", parameters=_SEARCH_PARAMS, function=_search_fn
            )

    def test_non_callable_function_rejected(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            OpenAITool(
                name="x", description="d", parameters=_SEARCH_PARAMS, function=42
            )

    def test_non_mapping_parameters_rejected(self) -> None:
        with pytest.raises(TypeError, match="JSON Schema mapping"):
            OpenAITool(
                name="x", description="d", parameters=[1, 2, 3], function=_search_fn
            )


# --------------------------------------------------------------------------- #
# Spec generation                                                             #
# --------------------------------------------------------------------------- #


class TestSpecs:
    def test_specs_match_openai_shape_and_preserve_order(self) -> None:
        tools = [
            OpenAITool("search", "search the web", _SEARCH_PARAMS, _search_fn),
            OpenAITool("send_email", "send mail", _SEND_PARAMS, _send_fn),
        ]
        specs = openai_tool_specs(tools)
        assert [s["function"]["name"] for s in specs] == ["search", "send_email"]
        assert all(s["type"] == "function" for s in specs)
        assert specs[0]["function"]["description"] == "search the web"
        assert specs[0]["function"]["parameters"] == _SEARCH_PARAMS

    def test_specs_are_independent_copies(self) -> None:
        tools = [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        specs = openai_tool_specs(tools)
        specs[0]["function"]["parameters"]["properties"]["query"]["type"] = "integer"
        # Source tool's parameters dict is untouched.
        assert _SEARCH_PARAMS["properties"]["query"]["type"] == "string"

    def test_prefix_namespaces_specs(self) -> None:
        tools = [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        specs = openai_tool_specs(tools, prefix="net")
        assert specs[0]["function"]["name"] == "net.search"

    def test_duplicate_names_in_specs_raise(self) -> None:
        tools = [
            OpenAITool("dup", "a", _SEARCH_PARAMS, _search_fn),
            OpenAITool("dup", "b", _SEARCH_PARAMS, _search_fn),
        ]
        with pytest.raises(ValueError, match="collision"):
            openai_tool_specs(tools)


# --------------------------------------------------------------------------- #
# Registration                                                                #
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_wrap_returns_callable_per_tool(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tools = [
            OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn),
            OpenAITool("send_email", "d", _SEND_PARAMS, _send_fn),
        ]
        wrapped = wrap_openai_tools(gw, tools)
        assert set(wrapped) == {"search", "send_email"}
        assert callable(wrapped["search"])

    def test_wrap_invokes_underlying_function_via_gateway(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        out = wrapped["search"](query="apg", limit=2)
        assert out == {"query": "apg", "results": ["r0", "r1"]}

    def test_wrap_registers_taint_spec_under_registered_name(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "search",
                    "d",
                    _SEARCH_PARAMS,
                    _search_fn,
                    taint_spec=ToolTaintSpec.of(adds=("web",)),
                )
            ],
        )
        assert "search" in gw.tool_specs
        assert gw.tool_specs["search"].adds == TaintLabel.of("web")

    def test_wrap_with_prefix_uses_prefixed_name_for_registration(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "search",
                    "d",
                    _SEARCH_PARAMS,
                    _search_fn,
                    taint_spec=ToolTaintSpec.of(adds=("web",)),
                )
            ],
            prefix="net",
        )
        assert "net.search" in gw.tool_specs
        assert "search" not in gw.tool_specs

    def test_duplicate_registration_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tools = [
            OpenAITool("dup", "a", _SEARCH_PARAMS, _search_fn),
            OpenAITool("dup", "b", _SEARCH_PARAMS, _search_fn),
        ]
        with pytest.raises(ValueError, match="collision"):
            wrap_openai_tools(gw, tools)


# --------------------------------------------------------------------------- #
# Dispatch — happy path                                                       #
# --------------------------------------------------------------------------- #


class TestDispatchAllow:
    def test_dispatch_allow_returns_role_tool_message_with_json_content(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        msg = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(name="search", arguments={"query": "apg", "limit": 1}),
        )
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert msg["name"] == "search"
        assert json.loads(msg["content"]) == {"query": "apg", "results": ["r0"]}

    def test_dispatch_supports_attribute_form_tool_call(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        tc = _AttrToolCall(
            id="call_42",
            function=_AttrFunction(name="search", arguments='{"query":"x"}'),
        )
        msg = dispatch_openai_tool_call(wrapped, tc)
        assert msg["tool_call_id"] == "call_42"
        assert json.loads(msg["content"])["query"] == "x"

    def test_dispatch_string_result_is_passed_through_verbatim(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "echo",
                    "d",
                    {"type": "object", "properties": {"x": {"type": "string"}}},
                    lambda x: x,
                )
            ],
        )
        msg = dispatch_openai_tool_call(
            wrapped, _make_tool_call(name="echo", arguments={"x": "hello"})
        )
        assert msg["content"] == "hello"

    def test_dispatch_forwards_input_label_to_gateway(self) -> None:
        captured: list[Any] = []

        def writer(call: ToolCall, decision: Any) -> None:
            captured.append((call, decision))

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "search",
                    "d",
                    _SEARCH_PARAMS,
                    _search_fn,
                    taint_spec=ToolTaintSpec.of(adds=("web",)),
                )
            ],
        )
        dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(name="search", arguments={"query": "x"}),
            input_label=TaintLabel.of("user"),
            agent_id="agent.research",
        )
        call, decision = captured[0]
        assert call.input_label == TaintLabel.of("user")
        assert call.agent_id == "agent.research"
        assert decision.output_label == TaintLabel.of("user", "web")

    def test_call_id_from_tool_call_propagates_to_audit_record(self) -> None:
        captured: list[ToolCall] = []

        def writer(call: ToolCall, _decision: Any) -> None:
            captured.append(call)

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        dispatch_openai_tool_call(
            wrapped, _make_tool_call(name="search", arguments={"query": "x"}, call_id="call_xyz")
        )
        assert captured[0].call_id == "call_xyz"


# --------------------------------------------------------------------------- #
# Dispatch — refusal                                                          #
# --------------------------------------------------------------------------- #


class TestDispatchRefusal:
    def test_dispatch_deny_returns_structured_tool_message(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email", reason="exfil risk")])
        send_calls: list[tuple[str, str]] = []

        def send(to: str, body: str) -> dict[str, Any]:
            send_calls.append((to, body))
            return {"ok": True}

        wrapped = wrap_openai_tools(
            gw, [OpenAITool("send_email", "d", _SEND_PARAMS, send)]
        )
        msg = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(name="send_email", arguments={"to": "a", "body": "b"}),
        )
        assert msg["role"] == "tool"
        body = json.loads(msg["content"])
        assert body["error"] == "policy_refusal"
        assert body["verdict"] == "deny"
        assert body["rule_id"] == "deny-by-name"
        assert body["reason"] == "exfil risk"
        # Underlying function was not called.
        assert send_calls == []

    def test_dispatch_review_returns_structured_tool_message(self) -> None:
        gw = Gateway(policies=[_review_named("search")])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        msg = dispatch_openai_tool_call(
            wrapped, _make_tool_call(name="search", arguments={"query": "x"})
        )
        body = json.loads(msg["content"])
        assert body["verdict"] == "review"
        assert body["rule_id"] == "review-by-name"

    def test_on_denied_callback_overrides_default_body(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email")])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("send_email", "d", _SEND_PARAMS, _send_fn)]
        )
        seen: list[Any] = []

        def on_denied(error: Any, tool_call: Any) -> str:
            seen.append((error.decision.verdict, tool_call["id"]))
            return "BLOCKED"

        msg = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(name="send_email", arguments={"to": "a", "body": "b"}),
            on_denied=on_denied,
        )
        assert msg["content"] == "BLOCKED"
        assert seen == [(Verdict.DENY, "call_1")]


# --------------------------------------------------------------------------- #
# Dispatch — error edges                                                      #
# --------------------------------------------------------------------------- #


class TestDispatchErrors:
    def test_unknown_tool_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        with pytest.raises(OpenAIToolCallError, match="unknown tool name"):
            dispatch_openai_tool_call(
                wrapped, _make_tool_call(name="unknown")
            )

    def test_missing_function_object_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        with pytest.raises(OpenAIToolCallError, match="'function'"):
            dispatch_openai_tool_call(wrapped, {"id": "call_1", "type": "function"})

    def test_empty_function_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        with pytest.raises(OpenAIToolCallError, match="non-empty string"):
            dispatch_openai_tool_call(
                wrapped,
                {"id": "call_1", "function": {"name": "", "arguments": "{}"}},
            )

    def test_missing_call_id_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        with pytest.raises(OpenAIToolCallError, match="id must be"):
            dispatch_openai_tool_call(
                wrapped,
                {"function": {"name": "search", "arguments": '{"query":"x"}'}},
            )

    def test_invalid_json_arguments_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        with pytest.raises(OpenAIToolCallError, match="not valid JSON"):
            dispatch_openai_tool_call(
                wrapped, _make_tool_call(name="search", arguments="not json")
            )

    def test_arguments_must_decode_to_object(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        with pytest.raises(OpenAIToolCallError, match="JSON object"):
            dispatch_openai_tool_call(
                wrapped, _make_tool_call(name="search", arguments="[1,2,3]")
            )

    def test_empty_arguments_string_treated_as_empty_object(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "noop",
                    "d",
                    {"type": "object", "properties": {}},
                    lambda: {"ok": True},
                )
            ],
        )
        msg = dispatch_openai_tool_call(
            wrapped, _make_tool_call(name="noop", arguments="")
        )
        assert json.loads(msg["content"]) == {"ok": True}

    def test_underlying_tool_exception_propagates(self) -> None:
        gw = Gateway(policies=[_allow_all()])

        def boom(**_kwargs: Any) -> None:
            raise RuntimeError("kapow")

        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "boom",
                    "d",
                    {"type": "object", "properties": {}},
                    boom,
                )
            ],
        )
        with pytest.raises(RuntimeError, match="kapow"):
            dispatch_openai_tool_call(
                wrapped, _make_tool_call(name="boom", arguments={})
            )

    def test_non_string_arguments_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_openai_tools(
            gw, [OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn)]
        )
        with pytest.raises(OpenAIToolCallError, match="must be a JSON string"):
            dispatch_openai_tool_call(
                wrapped,
                {
                    "id": "call_1",
                    "function": {"name": "search", "arguments": {"query": "x"}},
                },
            )


# --------------------------------------------------------------------------- #
# Dispatch — resource binding                                                 #
# --------------------------------------------------------------------------- #


class TestDispatchResource:
    def test_resource_arg_binds_value_for_selector_matching(self) -> None:
        policy = Policy(
            name="resource",
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
        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "send_email",
                    "d",
                    _SEND_PARAMS,
                    _send_fn,
                    resource_arg="to",
                )
            ],
        )

        denied = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(
                name="send_email", arguments={"to": "ops@example.com", "body": "b"}
            ),
        )
        assert json.loads(denied["content"])["error"] == "policy_refusal"

        allowed = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(
                name="send_email", arguments={"to": "alice@example.com", "body": "b"}
            ),
        )
        assert json.loads(allowed["content"])["sent_to"] == "alice@example.com"

    def test_apg_resource_override_via_dispatch_kwarg_takes_precedence(self) -> None:
        policy = Policy(
            name="resource-override",
            rules=(
                Rule(
                    id="deny-special",
                    when=Selector(tool="send_email", resource="forbidden"),
                    effect=Effect(action=Action.DENY, reason="nope"),
                ),
                Rule(id="allow-rest", when=Selector(), effect=Effect(action=Action.ALLOW)),
            ),
        )
        gw = Gateway(policies=[policy])
        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "send_email",
                    "d",
                    _SEND_PARAMS,
                    _send_fn,
                    resource_arg="to",
                )
            ],
        )
        msg = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(
                name="send_email", arguments={"to": "alice@example.com", "body": "b"}
            ),
            resource="forbidden",
        )
        body = json.loads(msg["content"])
        assert body["error"] == "policy_refusal"


# --------------------------------------------------------------------------- #
# Batched dispatch                                                            #
# --------------------------------------------------------------------------- #


class TestBatchDispatch:
    def test_dispatch_many_returns_one_message_per_call(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email")])
        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool("search", "d", _SEARCH_PARAMS, _search_fn),
                OpenAITool("send_email", "d", _SEND_PARAMS, _send_fn),
            ],
        )
        calls = [
            _make_tool_call(name="search", arguments={"query": "x"}, call_id="a"),
            _make_tool_call(
                name="send_email", arguments={"to": "x", "body": "y"}, call_id="b"
            ),
        ]
        msgs = dispatch_openai_tool_calls(wrapped, calls)
        assert len(msgs) == 2
        assert [m["tool_call_id"] for m in msgs] == ["a", "b"]
        assert json.loads(msgs[0]["content"])["query"] == "x"
        assert json.loads(msgs[1]["content"])["error"] == "policy_refusal"


# --------------------------------------------------------------------------- #
# Worked exfiltration scenario                                                #
# --------------------------------------------------------------------------- #


class TestExfiltrationScenario:
    def test_web_tainted_send_email_is_denied(self) -> None:
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
        wrapped = wrap_openai_tools(
            gw,
            [
                OpenAITool(
                    "search",
                    "d",
                    _SEARCH_PARAMS,
                    _search_fn,
                    taint_spec=ToolTaintSpec.of(adds=("web",)),
                ),
                OpenAITool(
                    "send_email",
                    "d",
                    _SEND_PARAMS,
                    _send_fn,
                ),
            ],
        )

        # 1) search() -> output carries 'web' taint.
        out = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(name="search", arguments={"query": "x"}, call_id="a"),
        )
        assert "results" in json.loads(out["content"])

        # 2) send_email() under web-tainted input is refused.
        denied = dispatch_openai_tool_call(
            wrapped,
            _make_tool_call(
                name="send_email",
                arguments={"to": "alice@example.com", "body": "b"},
                call_id="b",
            ),
            input_label=TaintLabel.of("web"),
        )
        body = json.loads(denied["content"])
        assert body["error"] == "policy_refusal"
        assert body["rule_id"] == "deny-web-send"
