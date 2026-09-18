"""Tests for the R59 model-in-the-loop runner (fakes only; no agentdojo)."""

from __future__ import annotations

from typing import Any

import pytest

from agent_policy_gateway.agentdojo_episodes import CallOutcome
from agent_policy_gateway.model_driver import ModelStep, ToolInvocation, ToolSchema, Usage
from agent_policy_gateway.model_loop import (
    DEFAULT_SYSTEM_PROMPT,
    ModelScenario,
    ModelScenarioReport,
    ModelTask,
    ModelTurnReport,
    aggregate_model_scenarios,
    read_model_scenarios,
    run_model_scenario,
    schemas_from_runtime,
    write_model_scenarios,
)


class FakeRuntime:
    """AgentDojo-shaped fake: refuses configured tools, executes the rest."""

    def __init__(self, refuse: set[str] | None = None) -> None:
        self.functions = {"read": object(), "act": object()}
        self.refuse = refuse or set()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.resets = 0

    def reset_taint(self) -> None:
        self.resets += 1

    @property
    def taint_label(self) -> Any:
        class _Label:
            all_sources = {"fake:src"} if self.calls else set()

        return _Label()

    def run_function(
        self, env: Any, function: str, kwargs: Any, raise_on_error: bool = False
    ) -> tuple[Any, str | None]:
        self.calls.append((function, dict(kwargs)))
        if function in self.refuse:
            return "", "PolicyDenied: refused by rule 'r1'"
        if isinstance(env, dict):
            env["ran"] = env.get("ran", []) + [function]
        return {"data": function}, None


class ScriptDriver:
    """Feeds a fixed step sequence and records what it was shown."""

    def __init__(self, steps: list[ModelStep]) -> None:
        self._steps = list(steps)
        self.seen: list[list[dict[str, Any]]] = []

    def next_step(self, messages: Any, tools: Any) -> ModelStep:
        self.seen.append([dict(m) for m in messages])
        return self._steps.pop(0)


def _scenario(n_tasks: int = 1) -> ModelScenario:
    return ModelScenario(
        scenario_id="s1",
        tasks=tuple(ModelTask(name=f"t{i}", prompt=f"do thing {i}") for i in range(n_tasks)),
        suite="fake",
        injection_task="inj",
    )


def test_scenario_validation() -> None:
    with pytest.raises(ValueError):
        ModelScenario(scenario_id="", tasks=())
    with pytest.raises(ValueError):
        ModelTask(name="", prompt="p")
    with pytest.raises(ValueError):
        ModelTask(name="t", prompt="")
    assert _scenario(3).horizon == 3


def test_run_model_scenario_executes_and_feeds_back() -> None:
    runtime = FakeRuntime()
    driver = ScriptDriver(
        [
            ModelStep(tool_calls=(ToolInvocation.of("read", {"k": 1}),), usage=Usage(5, 1)),
            ModelStep(text="all done", usage=Usage(7, 2)),
        ]
    )
    env: dict[str, Any] = {}
    report = run_model_scenario(driver, runtime, _scenario(), env=env, arm="a")
    assert runtime.resets == 1
    assert runtime.calls == [("read", {"k": 1})]
    assert env["ran"] == ["read"]
    turn = report.turns[0]
    assert [c.status for c in turn.calls] == ["executed"]
    assert turn.final_text == "all done"
    assert turn.steps == 2
    assert not turn.truncated
    assert turn.taint_after == ("fake:src",)
    assert turn.usage == Usage(input_tokens=12, output_tokens=3, estimated=False)
    # The driver's second request saw the system prompt, the task, its own
    # tool call, and the tool result.
    roles = [m["role"] for m in driver.seen[1]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert driver.seen[1][0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert driver.seen[1][3]["tool"] == "read"
    assert driver.seen[1][3]["error"] is None
    assert '"data": "read"' in driver.seen[1][3]["content"]


def test_refusal_is_classified_and_visible_to_driver() -> None:
    runtime = FakeRuntime(refuse={"act"})
    driver = ScriptDriver(
        [
            ModelStep(tool_calls=(ToolInvocation.of("act"),)),
            ModelStep(text="gave up"),
        ]
    )
    report = run_model_scenario(driver, runtime, _scenario())
    turn = report.turns[0]
    assert [c.status for c in turn.calls] == ["refused"]
    assert turn.refusals == 1
    assert driver.seen[1][3]["error"].startswith("PolicyDenied:")


def test_max_steps_truncates_turn() -> None:
    runtime = FakeRuntime()
    steps = [ModelStep(tool_calls=(ToolInvocation.of("read"),)) for _ in range(10)]
    driver = ScriptDriver(steps)
    report = run_model_scenario(driver, runtime, _scenario(), max_steps=3)
    turn = report.turns[0]
    assert turn.truncated
    assert turn.steps == 3
    assert turn.final_text == ""


def test_utility_and_security_judged_on_snapshots() -> None:
    runtime = FakeRuntime()
    driver = ScriptDriver(
        [
            ModelStep(tool_calls=(ToolInvocation.of("act"),)),
            ModelStep(text="turn one done"),
            ModelStep(text="turn two done"),
        ]
    )
    env: dict[str, Any] = {"ran": []}
    judged: list[tuple[str, str, Any, Any]] = []

    def utility_fn(name: str, output: str, pre: Any, post: Any) -> bool:
        judged.append((name, output, pre["ran"], post["ran"]))
        return name == "t0"

    security_seen: dict[str, Any] = {}

    def security_fn(output: str, pre: Any, post: Any) -> bool:
        security_seen.update(output=output, pre=pre["ran"], post=post["ran"])
        return True

    report = run_model_scenario(
        driver,
        runtime,
        _scenario(2),
        env=env,
        utility_fn=utility_fn,
        security_fn=security_fn,
    )
    # Turn snapshots: t0's pre-env is empty, its post-env has the call;
    # t1 sees the mutated env as its pre-env (the session persists).
    assert judged[0] == ("t0", "turn one done", [], ["act"])
    assert judged[1] == ("t1", "turn two done", ["act"], ["act"])
    # Scenario snapshots span the whole horizon.
    assert security_seen == {"output": "turn two done", "pre": [], "post": ["act"]}
    assert report.turns[0].utility is True
    assert report.turns[1].utility is False
    assert report.utility == 0.5
    assert report.security_breached is True


def test_retried_refusals_counts_adjacent_same_function() -> None:
    refused = CallOutcome(function="act", kind="model", status="refused", error="PolicyDenied: r")
    executed = CallOutcome(function="act", kind="model", status="executed")
    other = CallOutcome(function="read", kind="model", status="executed")
    turn = ModelTurnReport(name="t", calls=(refused, refused, executed, other, refused))
    # pairs: (refused, refused) and (refused, executed) match; trailing refusal has no successor.
    assert turn.retried_refusals == 2
    assert turn.refusals == 3
    assert turn.executed_calls == 2


def test_report_totals_and_serialization_round_trip(tmp_path: Any) -> None:
    runtime = FakeRuntime(refuse={"act"})
    driver = ScriptDriver(
        [
            ModelStep(tool_calls=(ToolInvocation.of("act"),), usage=Usage(5, 1, True)),
            ModelStep(text="done", usage=Usage(3, 1, False)),
        ]
    )
    report = run_model_scenario(
        driver, runtime, _scenario(), arm="apg/skip", security_fn=lambda o, a, b: False
    )
    assert report.total_refusals == 1
    assert report.total_steps == 2
    assert report.total_usage == Usage(input_tokens=8, output_tokens=2, estimated=True)
    assert report.security_breached is False
    d = report.to_dict()
    assert d["arm"] == "apg/skip"
    assert d["total_usage"]["estimated"] is True

    path = tmp_path / "reports.jsonl"
    assert write_model_scenarios([report], path) == 1
    (loaded,) = read_model_scenarios(path)
    assert loaded == d

    path.write_text("{bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_model_scenarios(path)


def test_aggregate_model_scenarios() -> None:
    def _report(utility: bool, breached: bool) -> ModelScenarioReport:
        turn = ModelTurnReport(
            name="t",
            calls=(CallOutcome(function="f", kind="model", status="executed"),),
            utility=utility,
            steps=2,
            usage=Usage(10, 5, True),
            wall_seconds=0.5,
        )
        return ModelScenarioReport(
            scenario_id="s", turns=(turn,), security_breached=breached
        )

    stats = aggregate_model_scenarios(
        [_report(True, False), _report(False, True)], arm="x"
    )
    assert stats.scenarios == 2
    assert stats.utility == 0.5
    assert stats.attack_success_rate == 0.5
    assert stats.driver_steps == 4
    assert stats.total_tokens == 30
    assert stats.tokens_estimated
    assert stats.wall_seconds == 1.0
    assert stats.to_dict()["utility"] == 0.5
    empty = aggregate_model_scenarios([], arm="e")
    assert empty.utility == 0.0
    assert empty.attack_success_rate == 0.0


def test_schemas_from_runtime_reads_descriptors_and_degrades() -> None:
    class _Params:
        @staticmethod
        def model_json_schema() -> dict[str, Any]:
            return {"type": "object", "properties": {"a": {"type": "integer"}}}

    class _Descriptor:
        description = "does a thing"
        parameters = _Params()

    class _Runtime:
        functions = {"rich": _Descriptor(), "bare": object()}

    schemas = {s.name: s for s in schemas_from_runtime(_Runtime())}
    assert schemas["rich"].description == "does a thing"
    assert schemas["rich"].parameters["properties"] == {"a": {"type": "integer"}}
    assert schemas["bare"].description == ""
    assert dict(schemas["bare"].parameters) == {}
    assert isinstance(schemas["rich"], ToolSchema)
