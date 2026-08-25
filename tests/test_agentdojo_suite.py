"""Tests for the AgentDojo task-suite wiring (R49b).

Three layers:

* Pure table/helper tests — the per-suite classification tables (untrusted
  readers, external sinks, resource args) and their accessor helpers, no
  ``agentdojo`` needed.
* ``policies/agentdojo.yaml`` — the shipped baseline policy is kept in
  lockstep with the sink tables: one deny rule per sink in the cross-suite
  union, each conditioned on ``taint.any_of: [agentdojo:untrusted]``, and
  behaviourally a tainted sink call is denied while untainted calls and
  tainted non-sink calls pass.
* ``gate_suite`` — wiring against a fake runtime (always), and against the
  real ``agentdojo`` banking suite plus the ``examples/agentdojo`` entry
  point when the benchmark is importable (skipped otherwise).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent_policy_gateway import (
    AGENTDOJO_SUITE_VERSION,
    AGENTDOJO_SUITES,
    AGENTDOJO_UNTRUSTED,
    Action,
    Gateway,
    Policy,
    TaintLabel,
    ToolCall,
    gate_suite,
    load_policy,
    suite_external_sinks,
    suite_resource_args,
    suite_taint_specs,
    suite_untrusted_readers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "policies" / "agentdojo.yaml"


def _sink_union() -> set[str]:
    union: set[str] = set()
    for suite in AGENTDOJO_SUITES:
        union |= suite_external_sinks(suite)
    return union


# --------------------------------------------------------------------------- #
# Classification tables                                                       #
# --------------------------------------------------------------------------- #


class TestTables:
    def test_covered_suites(self) -> None:
        assert AGENTDOJO_SUITES == ("banking", "slack", "travel", "workspace")

    @pytest.mark.parametrize("suite", AGENTDOJO_SUITES)
    def test_every_suite_has_readers_and_sinks(self, suite: str) -> None:
        assert suite_untrusted_readers(suite)
        assert suite_external_sinks(suite)

    @pytest.mark.parametrize("suite", AGENTDOJO_SUITES)
    def test_readers_and_sinks_disjoint(self, suite: str) -> None:
        overlap = suite_untrusted_readers(suite) & suite_external_sinks(suite)
        assert not overlap, f"{suite}: {overlap}"

    @pytest.mark.parametrize("suite", AGENTDOJO_SUITES)
    def test_resource_args_name_only_sinks(self, suite: str) -> None:
        extra = set(suite_resource_args(suite)) - suite_external_sinks(suite)
        assert not extra, f"{suite}: {extra}"

    @pytest.mark.parametrize("suite", AGENTDOJO_SUITES)
    def test_taint_specs_mark_every_reader_untrusted(self, suite: str) -> None:
        specs = suite_taint_specs(suite)
        assert set(specs) == set(suite_untrusted_readers(suite))
        for spec in specs.values():
            assert spec.adds == TaintLabel.of(AGENTDOJO_UNTRUSTED)
            assert spec.declassifies.is_empty()

    @pytest.mark.parametrize(
        "helper",
        [
            suite_untrusted_readers,
            suite_external_sinks,
            suite_taint_specs,
            suite_resource_args,
        ],
    )
    def test_unknown_suite_raises(self, helper: Any) -> None:
        with pytest.raises(ValueError, match="unknown AgentDojo suite"):
            helper("no-such-suite")

    def test_suite_version_pinned(self) -> None:
        assert AGENTDOJO_SUITE_VERSION == "v1.2.1"


# --------------------------------------------------------------------------- #
# policies/agentdojo.yaml                                                     #
# --------------------------------------------------------------------------- #


class TestBaselinePolicy:
    @pytest.fixture()
    def policy(self) -> Policy:
        return load_policy(POLICY_PATH)

    def test_one_deny_rule_per_sink_in_union(self, policy: Policy) -> None:
        rule_tools = [r.when.tool for r in policy.rules]
        assert sorted(rule_tools) == sorted(_sink_union())
        assert len(rule_tools) == len(set(rule_tools))

    def test_every_rule_denies_on_untrusted_taint(self, policy: Policy) -> None:
        for rule in policy.rules:
            assert rule.effect.action == Action.DENY, rule.id
            assert rule.id == f"deny-untrusted-to-{rule.when.tool}"
            assert rule.when.taint is not None, rule.id
            assert tuple(rule.when.taint.any_of) == (AGENTDOJO_UNTRUSTED,)
            assert not rule.when.taint.all_of and not rule.when.taint.none_of

    @pytest.mark.parametrize("sink", sorted({"send_money", "send_email", "post_webpage"}))
    def test_tainted_sink_call_is_denied(self, policy: Policy, sink: str) -> None:
        call = ToolCall(tool_name=sink, input_label=TaintLabel.of(AGENTDOJO_UNTRUSTED))
        rule = policy.first_match(call)
        assert rule is not None and rule.effect.action == Action.DENY

    def test_untainted_sink_call_matches_no_rule(self, policy: Policy) -> None:
        call = ToolCall(tool_name="send_money", input_label=TaintLabel())
        assert policy.first_match(call) is None

    def test_tainted_reader_call_matches_no_rule(self, policy: Policy) -> None:
        call = ToolCall(
            tool_name="get_most_recent_transactions",
            input_label=TaintLabel.of(AGENTDOJO_UNTRUSTED),
        )
        assert policy.first_match(call) is None


# --------------------------------------------------------------------------- #
# gate_suite against a fake runtime                                           #
# --------------------------------------------------------------------------- #


@dataclass
class _FakeRuntime:
    """Duck-typed stand-in for ``agentdojo.functions_runtime.FunctionsRuntime``."""

    functions: dict[str, Any] = field(default_factory=dict)

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: dict[str, Any],
        raise_on_error: bool = False,
    ) -> tuple[Any, str | None]:
        try:
            result = self.functions[function](**kwargs)
        except Exception as exc:
            if raise_on_error:
                raise
            return "", f"{type(exc).__name__}: {exc}"
        return result, None


def _fake_banking_runtime() -> _FakeRuntime:
    tools = suite_untrusted_readers("banking") | suite_external_sinks("banking")
    return _FakeRuntime(functions={name: (lambda **kw: "ok") for name in tools})


class TestGateSuiteFake:
    @pytest.fixture()
    def gated(self) -> Any:
        gateway = Gateway(policies=[load_policy(POLICY_PATH)])
        return gate_suite(gateway, "banking", runtime=_fake_banking_runtime())

    def test_suite_name_without_runtime_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="pass runtime="):
            gate_suite(Gateway(), "banking")

    def test_unknown_suite_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown AgentDojo suite"):
            gate_suite(Gateway(), "no-such-suite", runtime=_FakeRuntime())

    def test_missing_classified_tool_raises(self) -> None:
        runtime = _fake_banking_runtime()
        del runtime.functions["send_money"]
        with pytest.raises(ValueError, match="missing classified tools: send_money"):
            gate_suite(Gateway(), "banking", runtime=runtime)

    def test_default_agent_id_names_the_suite(self, gated: Any) -> None:
        assert gated._agent_id == "agentdojo:banking"

    def test_reader_taints_the_session(self, gated: Any) -> None:
        _, error = gated.run_function(None, "read_file", {"file_path": "bill.txt"})
        assert error is None
        assert gated.taint_label == TaintLabel.of(AGENTDOJO_UNTRUSTED)

    def test_untainted_sink_is_allowed(self, gated: Any) -> None:
        result, error = gated.run_function(
            None, "send_money", {"recipient": "UK1", "amount": 1.0}
        )
        assert error is None and result == "ok"

    def test_sink_after_reader_is_denied_with_rule_id(self, gated: Any) -> None:
        gated.run_function(None, "get_most_recent_transactions", {"n": 100})
        result, error = gated.run_function(
            None, "send_money", {"recipient": "US-attacker", "amount": 100.0}
        )
        assert result == ""
        assert error is not None
        assert error.startswith("PolicyDenied: refused by rule 'deny-untrusted-to-send_money'")

    def test_reset_taint_restores_the_sink(self, gated: Any) -> None:
        gated.run_function(None, "read_file", {"file_path": "bill.txt"})
        gated.reset_taint()
        _, error = gated.run_function(None, "send_money", {"recipient": "UK1"})
        assert error is None


# --------------------------------------------------------------------------- #
# Integration against the real benchmark (skipped without agentdojo)          #
# --------------------------------------------------------------------------- #


def _real_suite(name: str) -> Any:
    pytest.importorskip("agentdojo")
    from agentdojo.task_suite.load_suites import get_suite

    return get_suite(AGENTDOJO_SUITE_VERSION, name)


class TestRealAgentDojo:
    @pytest.mark.parametrize("suite_name", AGENTDOJO_SUITES)
    def test_tables_are_subsets_of_real_suite_tools(self, suite_name: str) -> None:
        suite = _real_suite(suite_name)
        real = {tool.name for tool in suite.tools}
        classified = suite_untrusted_readers(suite_name) | suite_external_sinks(suite_name)
        assert classified <= real, classified - real

    def test_gate_suite_builds_runtime_from_suite_tools(self) -> None:
        suite = _real_suite("banking")
        gated = gate_suite(Gateway(policies=[load_policy(POLICY_PATH)]), suite)
        assert set(gated.functions) == {tool.name for tool in suite.tools}

    def test_benign_task_passes_and_injection_is_blocked(self) -> None:
        _real_suite("banking")
        from examples.agentdojo import (
            attacker_was_paid,
            run_benign,
            run_injection,
        )

        benign = run_benign()
        assert benign.errors == []

        injection = run_injection()
        read_error = injection.steps[0][1]
        sink_error = injection.steps[1][1]
        assert read_error is None
        assert sink_error is not None
        assert sink_error.startswith(
            "PolicyDenied: refused by rule 'deny-untrusted-to-send_money'"
        )
        assert not attacker_was_paid(injection)

    def test_example_entry_point_exits_zero(self) -> None:
        pytest.importorskip("agentdojo")
        proc = subprocess.run(
            [sys.executable, "-m", "examples.agentdojo"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "injection: exfiltration blocked" in proc.stdout
