"""Tests for the R59 model-driver seam: types, digests, record/replay, Anthropic edge."""

from __future__ import annotations

from typing import Any

import pytest

from agent_policy_gateway.model_driver import (
    AnthropicDriver,
    ModelDriver,
    ModelStep,
    RecordingDriver,
    ReplayDriver,
    ReplayMismatch,
    ToolInvocation,
    ToolSchema,
    Usage,
    request_digest,
)

MESSAGES = [
    {"role": "system", "content": "be helpful"},
    {"role": "user", "content": "pay the bill"},
]
TOOLS = [ToolSchema(name="send_money", description="wire money", parameters={"type": "object"})]


class StubDriver:
    """Plays back a fixed sequence of steps."""

    def __init__(self, steps: list[ModelStep]) -> None:
        self._steps = list(steps)
        self.calls = 0

    def next_step(self, messages: Any, tools: Any) -> ModelStep:
        self.calls += 1
        return self._steps.pop(0)


# ----- value types ----------------------------------------------------------


def test_tool_invocation_round_trip() -> None:
    inv = ToolInvocation.of("send_money", {"amount": 9.5, "recipient": "GB1"})
    assert inv.arg_dict() == {"amount": 9.5, "recipient": "GB1"}
    assert ToolInvocation.from_dict(inv.to_dict()) == inv


def test_tool_invocation_requires_function() -> None:
    with pytest.raises(ValueError):
        ToolInvocation(function="")


def test_model_step_round_trip_and_finality() -> None:
    step = ModelStep(
        tool_calls=(ToolInvocation.of("f", {"a": 1}),),
        text="calling f",
        usage=Usage(input_tokens=10, output_tokens=3, estimated=True),
    )
    assert not step.is_final
    restored = ModelStep.from_dict(step.to_dict())
    assert restored == step
    assert ModelStep(text="done").is_final
    assert Usage(input_tokens=2, output_tokens=3).total_tokens == 5


def test_drivers_satisfy_protocol() -> None:
    assert isinstance(StubDriver([]), ModelDriver)


# ----- request digest -------------------------------------------------------


def test_request_digest_stable_and_sensitive() -> None:
    d1 = request_digest(MESSAGES, TOOLS)
    assert d1 == request_digest([dict(m) for m in MESSAGES], list(TOOLS))
    assert d1 != request_digest(MESSAGES + [{"role": "user", "content": "x"}], TOOLS)
    assert d1 != request_digest(MESSAGES, [ToolSchema(name="other")])


# ----- record / replay ------------------------------------------------------


def test_record_then_replay_round_trip(tmp_path: Any) -> None:
    steps = [
        ModelStep(tool_calls=(ToolInvocation.of("f", {"a": 1}),), usage=Usage(1, 2, True)),
        ModelStep(text="done"),
    ]
    path = tmp_path / "fixture.jsonl"
    recorder = RecordingDriver(StubDriver(list(steps)), path)
    later = MESSAGES + [{"role": "tool", "tool": "f", "content": "ok", "error": None}]
    assert recorder.next_step(MESSAGES, TOOLS) == steps[0]
    assert recorder.next_step(later, TOOLS) == steps[1]
    assert recorder.recorded == 2

    replay = ReplayDriver(path)
    assert len(replay) == 2
    assert replay.next_step(MESSAGES, TOOLS) == steps[0]
    assert replay.remaining == 1
    assert replay.next_step(later, TOOLS) == steps[1]
    assert replay.remaining == 0


def test_replay_strict_mismatch_raises(tmp_path: Any) -> None:
    path = tmp_path / "fixture.jsonl"
    RecordingDriver(StubDriver([ModelStep(text="done")]), path).next_step(MESSAGES, TOOLS)
    replay = ReplayDriver(path)
    with pytest.raises(ReplayMismatch, match="digest mismatch"):
        replay.next_step([{"role": "user", "content": "different"}], TOOLS)


def test_replay_non_strict_skips_digest_check(tmp_path: Any) -> None:
    path = tmp_path / "fixture.jsonl"
    RecordingDriver(StubDriver([ModelStep(text="done")]), path).next_step(MESSAGES, TOOLS)
    replay = ReplayDriver(path, strict=False)
    assert replay.next_step([{"role": "user", "content": "different"}], TOOLS).text == "done"


def test_replay_exhausted_raises(tmp_path: Any) -> None:
    path = tmp_path / "fixture.jsonl"
    RecordingDriver(StubDriver([ModelStep(text="done")]), path).next_step(MESSAGES, TOOLS)
    replay = ReplayDriver(path)
    replay.next_step(MESSAGES, TOOLS)
    with pytest.raises(ReplayMismatch, match="exhausted"):
        replay.next_step(MESSAGES, TOOLS)


def test_replay_malformed_fixture_raises(tmp_path: Any) -> None:
    path = tmp_path / "fixture.jsonl"
    path.write_text('{"digest": "x"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        ReplayDriver(path)


# ----- Anthropic driver (fake client; no key, no network) -------------------


class _FakeBlock:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


class _FakeResponse:
    def __init__(self, content: list[Any], usage: Any) -> None:
        self.content = content
        self.usage = usage


class _FakeMessagesAPI:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.last_request: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last_request = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.messages = _FakeMessagesAPI(response)


def _fake_driver(response: _FakeResponse) -> tuple[AnthropicDriver, _FakeClient]:
    client = _FakeClient(response)
    return AnthropicDriver("test-model", client=client), client


def test_anthropic_driver_parses_tool_use_and_usage() -> None:
    response = _FakeResponse(
        content=[
            _FakeBlock(type="text", text="paying now"),
            _FakeBlock(type="tool_use", name="send_money", input={"amount": 5}),
        ],
        usage=_FakeBlock(input_tokens=100, output_tokens=20),
    )
    driver, _ = _fake_driver(response)
    step = driver.next_step(MESSAGES, TOOLS)
    assert step.tool_calls == (ToolInvocation.of("send_money", {"amount": 5}),)
    assert step.text == "paying now"
    assert step.usage == Usage(input_tokens=100, output_tokens=20, estimated=False)


def test_anthropic_driver_converts_neutral_messages() -> None:
    response = _FakeResponse(content=[], usage=None)
    driver, client = _fake_driver(response)
    driver.next_step(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "on it",
                "tool_calls": [{"call_id": "c1", "function": "f", "args": {"a": 1}}],
            },
            {
                "role": "tool",
                "tool": "f",
                "call_id": "c1",
                "content": "",
                "error": "PolicyDenied: no",
            },
        ],
        TOOLS,
    )
    request = client.messages.last_request
    assert request is not None
    assert request["system"] == "sys"
    assert request["messages"][0] == {"role": "user", "content": "task"}
    assistant = request["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "on it"}
    assert assistant["content"][1]["type"] == "tool_use"
    assert assistant["content"][1]["input"] == {"a": 1}
    tool_result = request["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["is_error"] is True
    assert "PolicyDenied" in tool_result["content"]
    assert request["tools"][0]["name"] == "send_money"
    assert request["tools"][0]["input_schema"] == {"type": "object"}


def test_anthropic_driver_requires_key_without_client(monkeypatch: Any) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        AnthropicDriver("test-model")
