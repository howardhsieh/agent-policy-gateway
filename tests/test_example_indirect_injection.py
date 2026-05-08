"""Tests for the worked indirect-injection example (R10).

The acceptance criteria for the example, restated as test invariants:

* The naive runner exfiltrates: ``send_email`` is invoked, the outbox
  contains the secret, and the recipient is the attacker.
* The gated runner refuses: the outbox stays empty, the agent reports a
  policy refusal, and the refusal carries the ``deny-web-to-email`` rule
  id from ``policies/default.yaml``.
* The audit log captures the refusal (and the upstream allow decisions
  for ``web_fetch`` and ``kb_lookup``) so an incident responder can
  reconstruct what the agent tried to do.
* The single-command entry point exits 0 and prints a line describing
  each variant's outcome.

The example imports ``agent_policy_gateway`` at the top level, so a broken
adapter or policy loader would surface as an import error here too.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent_policy_gateway import Verdict
from examples.indirect_injection import (
    ATTACKER_EMAIL,
    INJECTED_PAGE_URL,
    run_gated,
    run_naive,
)
from examples.indirect_injection.agent import run_research_task
from examples.indirect_injection.gated import build_gateway
from examples.indirect_injection.injected_page import INJECTED_PAGE_HTML
from examples.indirect_injection.tools import SECRET_VALUE, build_tools

# ----- naive runner --------------------------------------------------------


def test_naive_runner_exfiltrates_to_attacker():
    result = run_naive()

    # The agent saw the injection and acted on it.
    assert result.agent.followed_injection
    assert result.agent.exfil_target == ATTACKER_EMAIL
    assert result.agent.exfil_record == "launch-codes"
    assert result.agent.refused_by_gateway is False
    assert result.agent.refusal_rule_id is None

    # The email was actually sent.
    assert len(result.outbox.sent) == 1
    sent = result.outbox.sent[0]
    assert sent.to == ATTACKER_EMAIL
    assert sent.body == SECRET_VALUE


def test_naive_runner_returns_summary_from_page_first_line():
    result = run_naive()
    # First line of the poisoned page is the benign-looking news header.
    assert result.agent.summary == INJECTED_PAGE_HTML.splitlines()[0]


# ----- gated runner --------------------------------------------------------


def test_gated_runner_refuses_send_email_and_outbox_stays_empty():
    result = run_gated()

    # The agent tried to follow the injection.
    assert result.agent.followed_injection
    assert result.agent.exfil_target == ATTACKER_EMAIL
    assert result.agent.exfil_record == "launch-codes"

    # The gateway refused it.
    assert result.agent.refused_by_gateway
    assert result.agent.refusal_rule_id == "deny-web-to-email"

    # No mail left the system.
    assert result.outbox.sent == []


def test_gated_runner_audit_log_captures_deny_decision():
    result = run_gated()

    verdicts = [decision.verdict for _, decision in result.audit]
    rule_ids = [decision.rule_id for _, decision in result.audit]

    # The web_fetch and kb_lookup calls were both allowed; the send_email
    # call was the one denied.
    assert Verdict.DENY in verdicts
    assert "deny-web-to-email" in rule_ids

    deny_call, deny_decision = next(
        (c, d) for c, d in result.audit if d.verdict == Verdict.DENY
    )
    assert deny_call.tool_name == "send_email"
    assert deny_call.args["to"] == ATTACKER_EMAIL
    assert deny_call.args["body"] == SECRET_VALUE
    assert deny_decision.rule_id == "deny-web-to-email"
    # The output label still includes the web source so callers can see
    # why it was denied.
    assert "web" in deny_decision.output_label.sources


def test_gated_runner_agent_call_ids_recorded_in_audit():
    result = run_gated()
    for call, _ in result.audit:
        assert call.agent_id == "agent.research"
        assert call.call_id  # non-empty


def test_build_gateway_uses_default_policy():
    gw = build_gateway()
    assert len(gw.policies) == 1
    assert gw.policies[0].name == "default"
    rule_ids = {r.id for r in gw.policies[0].rules}
    assert "deny-web-to-email" in rule_ids


# ----- injection-content invariants ---------------------------------------


def test_injected_page_contains_attacker_address_and_record_name():
    # If somebody refactors the page text in a way that breaks the demo's
    # injection pattern, this test will catch it before the runner does.
    assert ATTACKER_EMAIL in INJECTED_PAGE_HTML
    assert "launch-codes" in INJECTED_PAGE_HTML
    assert "send its contents to" in INJECTED_PAGE_HTML.lower()


def test_unknown_url_raises_in_demo_web_fetch():
    tools = build_tools()
    with pytest.raises(ValueError):
        tools.web_fetch("https://other.example/")


# ----- driver: agent stub behaves the same when wrapped or not ------------


def test_agent_driver_no_injection_returns_clean_result():
    """If the page contains no injection, the agent does not call send_email."""
    tools = build_tools()
    page = "Quarterly Update\n\nNothing to see here."

    def web_fetch(_url: str) -> str:
        return page

    sent: list[tuple[str, str]] = []

    def send_email(to: str, body: str) -> dict:
        sent.append((to, body))
        return {"to": to, "delivered": True}

    result = run_research_task(
        web_fetch=web_fetch,
        kb_lookup=tools.kb_lookup,
        send_email=send_email,
        page_url=INJECTED_PAGE_URL,
    )
    assert result.followed_injection is False
    assert result.exfil_target is None
    assert sent == []


# ----- entry point --------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_main_module_runs_and_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "examples.indirect_injection"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "naive: exfiltrated" in proc.stdout
    assert "gated: refused by rule 'deny-web-to-email'" in proc.stdout
