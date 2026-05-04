"""Tests for the Anthropic Messages-API tool-use adapter (R8).

The adapter exposes :func:`anthropic_tool_specs` (build the JSON tool
descriptors sent in the API request) and :func:`wrap_anthropic_tools`
(mount Python tools under a :class:`Gateway` and return a name -> callable
map), with :func:`dispatch_anthropic_tool_use` translating model-produced
``tool_use`` content blocks into gateway-mediated invocations and
``tool_result`` content blocks. These tests exercise the adapter
end-to-end against ad-hoc fakes — the real ``anthropic`` SDK is
intentionally *not* a dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from agent_policy_gateway import (
    Action,
    AnthropicTool,
    AnthropicToolUseError,
    Effect,
    Gateway,
    Policy,
    Rule,
    Selector,
    TaintCondition,
    TaintLabel,
    ToolCall,
    ToolTaintSpec,
    Verdict,
    anthropic_tool_specs,
    dispatch_anthropic_tool_use,
    dispatch_anthropic_tool_uses,
    wrap_anthropic_tools,
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


def _make_tool_use(
    *,
    name: str,
    input: dict[str, Any] | str | None = None,
    use_id: str = "toolu_1",
    include_type: bool = True,
) -> dict[str, Any]:
    """Build a dict-form ``tool_use`` content block for tests.

    ``input`` is passed through as-is (Anthropic emits a JSON object for
    ``input``, but the adapter also tolerates a JSON-encoded string for
    robustness — both shapes get exercised in the tests).
    """
    block: dict[str, Any] = {
        "id": use_id,
        "name": name,
        "input": {} if input is None else input,
    }
    if include_type:
        block["type"] = "tool_use"
    return block


@dataclass
class _AttrToolUse:
    """Mimics the anthropic-python SDK's pydantic-style ToolUseBlock."""

    id: str
    name: str
    input: Any
    type: str = "tool_use"


# Sample Python tools used across the test cases.

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["query"],
}

_SEND_SCHEMA = {
    "type": "object",
    "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
    "required": ["to", "body"],
}


def _search_fn(query: str, limit: int = 5) -> dict[str, Any]:
    return {"query": query, "results": [f"r{i}" for i in range(limit)]}


def _send_fn(to: str, body: str) -> dict[str, Any]:
    return {"sent_to": to, "size": len(body)}


# --------------------------------------------------------------------------- #
# AnthropicTool construction                                                  #
# --------------------------------------------------------------------------- #


class TestAnthropicToolConstruction:
    def test_minimal_construction(self) -> None:
        t = AnthropicTool(
            name="search",
            description="d",
            input_schema=_SEARCH_SCHEMA,
            function=_search_fn,
        )
        assert t.name == "search"
        assert t.resource_arg is None
        assert t.taint_spec is None

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            AnthropicTool(
                name="",
                description="d",
                input_schema=_SEARCH_SCHEMA,
                function=_search_fn,
            )

    def test_non_callable_function_rejected(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            AnthropicTool(
                name="x",
                description="d",
                input_schema=_SEARCH_SCHEMA,
                function=42,
            )

    def test_non_mapping_input_schema_rejected(self) -> None:
        with pytest.raises(TypeError, match="JSON Schema mapping"):
            AnthropicTool(
                name="x",
                description="d",
                input_schema=[1, 2, 3],
                function=_search_fn,
            )


# --------------------------------------------------------------------------- #
# Spec generation                                                             #
# --------------------------------------------------------------------------- #


class TestSpecs:
    def test_specs_match_anthropic_shape_and_preserve_order(self) -> None:
        tools = [
            AnthropicTool("search", "search the web", _SEARCH_SCHEMA, _search_fn),
            AnthropicTool("send_email", "send mail", _SEND_SCHEMA, _send_fn),
        ]
        specs = anthropic_tool_specs(tools)
        assert [s["name"] for s in specs] == ["search", "send_email"]
        # Anthropic Messages API uses a flat shape — no "type"/"function" wrapper.
        for s in specs:
            assert "type" not in s
            assert "function" not in s
            assert set(s) == {"name", "description", "input_schema"}
        assert specs[0]["description"] == "search the web"
        assert specs[0]["input_schema"] == _SEARCH_SCHEMA

    def test_specs_are_independent_copies(self) -> None:
        tools = [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        specs = anthropic_tool_specs(tools)
        specs[0]["input_schema"]["properties"]["query"]["type"] = "integer"
        # Source tool's schema dict is untouched.
        assert _SEARCH_SCHEMA["properties"]["query"]["type"] == "string"

    def test_prefix_namespaces_specs(self) -> None:
        tools = [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        specs = anthropic_tool_specs(tools, prefix="net")
        assert specs[0]["name"] == "net.search"

    def test_duplicate_names_in_specs_raise(self) -> None:
        tools = [
            AnthropicTool("dup", "a", _SEARCH_SCHEMA, _search_fn),
            AnthropicTool("dup", "b", _SEARCH_SCHEMA, _search_fn),
        ]
        with pytest.raises(ValueError, match="collision"):
            anthropic_tool_specs(tools)


# --------------------------------------------------------------------------- #
# Registration                                                                #
# --------------------------------------------------------------------------- #


class TestRegistration:
    def test_wrap_returns_callable_per_tool(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        tools = [
            AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn),
            AnthropicTool("send_email", "d", _SEND_SCHEMA, _send_fn),
        ]
        wrapped = wrap_anthropic_tools(gw, tools)
        assert set(wrapped) == {"search", "send_email"}
        assert callable(wrapped["search"])

    def test_wrap_invokes_underlying_function_via_gateway(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        out = wrapped["search"](query="apg", limit=2)
        assert out == {"query": "apg", "results": ["r0", "r1"]}

    def test_wrap_registers_taint_spec_under_registered_name(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "search",
                    "d",
                    _SEARCH_SCHEMA,
                    _search_fn,
                    taint_spec=ToolTaintSpec.of(adds=("web",)),
                )
            ],
        )
        assert "search" in gw.tool_specs
        assert gw.tool_specs["search"].adds == TaintLabel.of("web")

    def test_wrap_with_prefix_uses_prefixed_name_for_registration(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "search",
                    "d",
                    _SEARCH_SCHEMA,
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
            AnthropicTool("dup", "a", _SEARCH_SCHEMA, _search_fn),
            AnthropicTool("dup", "b", _SEARCH_SCHEMA, _search_fn),
        ]
        with pytest.raises(ValueError, match="collision"):
            wrap_anthropic_tools(gw, tools)


# --------------------------------------------------------------------------- #
# Dispatch — happy path                                                       #
# --------------------------------------------------------------------------- #


class TestDispatchAllow:
    def test_dispatch_allow_returns_tool_result_block_with_json_content(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        block = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(name="search", input={"query": "apg", "limit": 1}),
        )
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "toolu_1"
        # is_error is omitted on allow (Anthropic treats absence as False).
        assert "is_error" not in block
        assert json.loads(block["content"]) == {"query": "apg", "results": ["r0"]}

    def test_dispatch_supports_attribute_form_tool_use(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        tu = _AttrToolUse(id="toolu_42", name="search", input={"query": "x"})
        block = dispatch_anthropic_tool_use(wrapped, tu)
        assert block["tool_use_id"] == "toolu_42"
        assert json.loads(block["content"])["query"] == "x"

    def test_dispatch_string_result_is_passed_through_verbatim(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "echo",
                    "d",
                    {"type": "object", "properties": {"x": {"type": "string"}}},
                    lambda x: x,
                )
            ],
        )
        block = dispatch_anthropic_tool_use(
            wrapped, _make_tool_use(name="echo", input={"x": "hello"})
        )
        assert block["content"] == "hello"

    def test_dispatch_input_as_json_string_is_tolerated(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        block = dispatch_anthropic_tool_use(
            wrapped, _make_tool_use(name="search", input='{"query":"x","limit":1}')
        )
        assert json.loads(block["content"]) == {"query": "x", "results": ["r0"]}

    def test_dispatch_forwards_input_label_to_gateway(self) -> None:
        captured: list[Any] = []

        def writer(call: ToolCall, decision: Any) -> None:
            captured.append((call, decision))

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "search",
                    "d",
                    _SEARCH_SCHEMA,
                    _search_fn,
                    taint_spec=ToolTaintSpec.of(adds=("web",)),
                )
            ],
        )
        dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(name="search", input={"query": "x"}),
            input_label=TaintLabel.of("user"),
            agent_id="agent.research",
        )
        call, decision = captured[0]
        assert call.input_label == TaintLabel.of("user")
        assert call.agent_id == "agent.research"
        assert decision.output_label == TaintLabel.of("user", "web")

    def test_use_id_propagates_to_audit_record_call_id(self) -> None:
        captured: list[ToolCall] = []

        def writer(call: ToolCall, _decision: Any) -> None:
            captured.append(call)

        gw = Gateway(policies=[_allow_all()], audit_writer=writer)
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(name="search", input={"query": "x"}, use_id="toolu_xyz"),
        )
        assert captured[0].call_id == "toolu_xyz"

    def test_dispatch_omits_type_field_in_tool_use_is_tolerated(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        block = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(name="search", input={"query": "x"}, include_type=False),
        )
        assert block["type"] == "tool_result"


# --------------------------------------------------------------------------- #
# Dispatch — refusal                                                          #
# --------------------------------------------------------------------------- #


class TestDispatchRefusal:
    def test_dispatch_deny_returns_error_block_with_structured_payload(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email", reason="exfil risk")])
        send_calls: list[tuple[str, str]] = []

        def send(to: str, body: str) -> dict[str, Any]:
            send_calls.append((to, body))
            return {"ok": True}

        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("send_email", "d", _SEND_SCHEMA, send)]
        )
        block = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(name="send_email", input={"to": "a", "body": "b"}),
        )
        assert block["type"] == "tool_result"
        assert block["is_error"] is True
        body = json.loads(block["content"])
        assert body["error"] == "policy_refusal"
        assert body["verdict"] == "deny"
        assert body["rule_id"] == "deny-by-name"
        assert body["reason"] == "exfil risk"
        # Underlying function was not called.
        assert send_calls == []

    def test_dispatch_review_returns_error_block_with_structured_payload(self) -> None:
        gw = Gateway(policies=[_review_named("search")])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        block = dispatch_anthropic_tool_use(
            wrapped, _make_tool_use(name="search", input={"query": "x"})
        )
        assert block["is_error"] is True
        body = json.loads(block["content"])
        assert body["verdict"] == "review"
        assert body["rule_id"] == "review-by-name"

    def test_on_denied_callback_overrides_default_body(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email")])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("send_email", "d", _SEND_SCHEMA, _send_fn)]
        )
        seen: list[Any] = []

        def on_denied(error: Any, tool_use: Any) -> str:
            seen.append((error.decision.verdict, tool_use["id"]))
            return "BLOCKED"

        block = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(name="send_email", input={"to": "a", "body": "b"}),
            on_denied=on_denied,
        )
        assert block["content"] == "BLOCKED"
        assert block["is_error"] is True
        assert seen == [(Verdict.DENY, "toolu_1")]


# --------------------------------------------------------------------------- #
# Dispatch — error edges                                                      #
# --------------------------------------------------------------------------- #


class TestDispatchErrors:
    def test_unknown_tool_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        with pytest.raises(AnthropicToolUseError, match="unknown tool name"):
            dispatch_anthropic_tool_use(wrapped, _make_tool_use(name="unknown"))

    def test_wrong_block_type_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        with pytest.raises(AnthropicToolUseError, match="tool_use"):
            dispatch_anthropic_tool_use(
                wrapped,
                {"type": "text", "id": "toolu_1", "name": "search", "input": {}},
            )

    def test_empty_tool_name_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        with pytest.raises(AnthropicToolUseError, match="name must be"):
            dispatch_anthropic_tool_use(
                wrapped,
                {"type": "tool_use", "id": "toolu_1", "name": "", "input": {}},
            )

    def test_missing_use_id_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        with pytest.raises(AnthropicToolUseError, match="id must be"):
            dispatch_anthropic_tool_use(
                wrapped,
                {"type": "tool_use", "name": "search", "input": {"query": "x"}},
            )

    def test_invalid_json_string_input_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        with pytest.raises(AnthropicToolUseError, match="not valid JSON"):
            dispatch_anthropic_tool_use(
                wrapped, _make_tool_use(name="search", input="not json")
            )

    def test_input_string_must_decode_to_object(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        with pytest.raises(AnthropicToolUseError, match="JSON object"):
            dispatch_anthropic_tool_use(
                wrapped, _make_tool_use(name="search", input="[1,2,3]")
            )

    def test_input_of_wrong_type_raises(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw, [AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn)]
        )
        with pytest.raises(AnthropicToolUseError, match="mapping or JSON string"):
            dispatch_anthropic_tool_use(
                wrapped, _make_tool_use(name="search", input=12345)
            )

    def test_empty_input_treated_as_empty_object(self) -> None:
        gw = Gateway(policies=[_allow_all()])
        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "noop",
                    "d",
                    {"type": "object", "properties": {}},
                    lambda: {"ok": True},
                )
            ],
        )
        block_none = dispatch_anthropic_tool_use(
            wrapped, _make_tool_use(name="noop", input=None)
        )
        block_empty = dispatch_anthropic_tool_use(
            wrapped, _make_tool_use(name="noop", input="")
        )
        for block in (block_none, block_empty):
            assert json.loads(block["content"]) == {"ok": True}

    def test_underlying_tool_exception_propagates(self) -> None:
        gw = Gateway(policies=[_allow_all()])

        def boom(**_kwargs: Any) -> None:
            raise RuntimeError("kapow")

        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "boom",
                    "d",
                    {"type": "object", "properties": {}},
                    boom,
                )
            ],
        )
        with pytest.raises(RuntimeError, match="kapow"):
            dispatch_anthropic_tool_use(
                wrapped, _make_tool_use(name="boom", input={})
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
        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "send_email",
                    "d",
                    _SEND_SCHEMA,
                    _send_fn,
                    resource_arg="to",
                )
            ],
        )

        denied = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(
                name="send_email", input={"to": "ops@example.com", "body": "b"}
            ),
        )
        assert denied["is_error"] is True
        assert json.loads(denied["content"])["error"] == "policy_refusal"

        allowed = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(
                name="send_email", input={"to": "alice@example.com", "body": "b"}
            ),
        )
        assert "is_error" not in allowed
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
        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "send_email",
                    "d",
                    _SEND_SCHEMA,
                    _send_fn,
                    resource_arg="to",
                )
            ],
        )
        block = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(
                name="send_email", input={"to": "alice@example.com", "body": "b"}
            ),
            resource="forbidden",
        )
        assert block["is_error"] is True
        body = json.loads(block["content"])
        assert body["error"] == "policy_refusal"


# --------------------------------------------------------------------------- #
# Batched dispatch                                                            #
# --------------------------------------------------------------------------- #


class TestBatchDispatch:
    def test_dispatch_many_returns_one_block_per_use(self) -> None:
        gw = Gateway(policies=[_deny_named("send_email")])
        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool("search", "d", _SEARCH_SCHEMA, _search_fn),
                AnthropicTool("send_email", "d", _SEND_SCHEMA, _send_fn),
            ],
        )
        uses = [
            _make_tool_use(name="search", input={"query": "x"}, use_id="toolu_a"),
            _make_tool_use(
                name="send_email", input={"to": "x", "body": "y"}, use_id="toolu_b"
            ),
        ]
        blocks = dispatch_anthropic_tool_uses(wrapped, uses)
        assert len(blocks) == 2
        assert [b["tool_use_id"] for b in blocks] == ["toolu_a", "toolu_b"]
        assert "is_error" not in blocks[0]
        assert blocks[1]["is_error"] is True
        assert json.loads(blocks[0]["content"])["query"] == "x"
        assert json.loads(blocks[1]["content"])["error"] == "policy_refusal"


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
        wrapped = wrap_anthropic_tools(
            gw,
            [
                AnthropicTool(
                    "search",
                    "d",
                    _SEARCH_SCHEMA,
                    _search_fn,
                    taint_spec=ToolTaintSpec.of(adds=("web",)),
                ),
                AnthropicTool(
                    "send_email",
                    "d",
                    _SEND_SCHEMA,
                    _send_fn,
                ),
            ],
        )

        # 1) search() -> output carries 'web' taint.
        out = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(name="search", input={"query": "x"}, use_id="toolu_a"),
        )
        assert "is_error" not in out
        assert "results" in json.loads(out["content"])

        # 2) send_email() under web-tainted input is refused.
        denied = dispatch_anthropic_tool_use(
            wrapped,
            _make_tool_use(
                name="send_email",
                input={"to": "alice@example.com", "body": "b"},
                use_id="toolu_b",
            ),
            input_label=TaintLabel.of("web"),
        )
        assert denied["is_error"] is True
        body = json.loads(denied["content"])
        assert body["error"] == "policy_refusal"
        assert body["rule_id"] == "deny-web-send"
