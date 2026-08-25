"""AgentDojo task-suite wiring for agent-policy-gateway (R49b).

The R49a adapter (:mod:`agent_policy_gateway.agentdojo_adapter`) mediates an
arbitrary AgentDojo ``FunctionsRuntime`` but leaves *what to declare about
each tool* to the caller. This module supplies those declarations for the
four default AgentDojo task suites (benchmark version
:data:`AGENTDOJO_SUITE_VERSION`) and a one-call entry point::

    from agentdojo.task_suite.load_suites import get_suite
    from agent_policy_gateway import Gateway, gate_suite, load_policy

    suite = get_suite("v1.2.1", "banking")
    gateway = Gateway(policies=[load_policy("policies/agentdojo.yaml")])
    gated = gate_suite(gateway, suite)          # a GatedAgentDojoRuntime
    result, error = gated.run_function(env, "send_money", {...})

Per suite, two disjoint tool sets are classified:

* **Untrusted readers** — tools whose *return value* can carry
  attacker-authored text. The sets are derived from where each suite's
  injection vectors live in its environment data (``data/suites/*``):
  banking places vectors in files and incoming-transaction subjects, slack
  in webpages and an externally-created channel name, travel in
  hotel/restaurant/car-rental reviews, and workspace in calendar event
  bodies, cloud-drive files, and received emails. Every reader gets a
  :class:`ToolTaintSpec` adding the :data:`AGENTDOJO_UNTRUSTED` source, so
  after the model reads any of them the session label (R49a's cumulative
  taint) is marked untrusted.
* **External sinks** — tools with outward-visible or privileged effects
  (money movement, outbound messages/email, credential and account
  mutation, sharing, deletion, reservations). These are the deny surface:
  ``policies/agentdojo.yaml`` carries one deny rule per sink in the
  cross-suite union, conditioned on ``taint.any_of: [agentdojo:untrusted]``,
  and the tables here are the source of truth those rules are checked
  against in the test suite.

The classification is deliberately *data*, not behaviour: the frozensets
below are the policy knob R50 measures, and alternative classifications can
be passed straight to :func:`wrap_agentdojo_runtime` without touching this
module. Like the R49a adapter, nothing here imports ``agentdojo`` at module
scope — :func:`gate_suite` lazy-imports ``FunctionsRuntime`` only when it
must build a runtime itself, so the package still ships without the
benchmark installed.
"""

from __future__ import annotations

from typing import Any

from agent_policy_gateway.agentdojo_adapter import (
    GatedAgentDojoRuntime,
    wrap_agentdojo_runtime,
)
from agent_policy_gateway.gateway import Gateway
from agent_policy_gateway.taint import ToolTaintSpec

#: Taint source added by every untrusted-content reader.
AGENTDOJO_UNTRUSTED = "agentdojo:untrusted"

#: The AgentDojo benchmark version the tables below were derived against.
AGENTDOJO_SUITE_VERSION = "v1.2.1"

#: The default task suites covered by the classification tables.
AGENTDOJO_SUITES = ("banking", "slack", "travel", "workspace")


_UNTRUSTED_READERS: dict[str, frozenset[str]] = {
    "banking": frozenset(
        {
            "read_file",  # bill / landlord-notice / address-change files
            "get_most_recent_transactions",  # injected incoming-transaction subject
        }
    ),
    "slack": frozenset(
        {
            "get_webpage",  # injected external webpages and blogs
            "get_channels",  # externally-created channel name is a vector
            "read_channel_messages",  # bodies authored by other workspace users
            "read_inbox",  # direct messages from other workspace users
        }
    ),
    "travel": frozenset(
        {
            "get_rating_reviews_for_hotels",
            "get_rating_reviews_for_restaurants",
            "get_rating_reviews_for_car_rental",
        }
    ),
    "workspace": frozenset(
        {
            "get_received_emails",
            "get_unread_emails",
            "search_emails",
            "get_day_calendar_events",
            "search_calendar_events",
            "get_file_by_id",
            "list_files",
            "search_files",
            "search_files_by_filename",
        }
    ),
}

_EXTERNAL_SINKS: dict[str, frozenset[str]] = {
    "banking": frozenset(
        {
            "send_money",
            "schedule_transaction",
            "update_scheduled_transaction",
            "update_password",
            "update_user_info",
        }
    ),
    "slack": frozenset(
        {
            "send_direct_message",
            "send_channel_message",
            "post_webpage",
            "invite_user_to_slack",
            "add_user_to_channel",
            "remove_user_from_slack",
        }
    ),
    "travel": frozenset(
        {
            "send_email",
            "reserve_hotel",
            "reserve_restaurant",
            "reserve_car_rental",
            "create_calendar_event",
            "cancel_calendar_event",
        }
    ),
    "workspace": frozenset(
        {
            "send_email",
            "share_file",
            "add_calendar_event_participants",
            "create_calendar_event",
            "reschedule_calendar_event",
            "cancel_calendar_event",
            "delete_email",
            "delete_file",
        }
    ),
}

_RESOURCE_ARGS: dict[str, dict[str, str]] = {
    "banking": {
        "send_money": "recipient",
        "schedule_transaction": "recipient",
        "update_scheduled_transaction": "recipient",
    },
    "slack": {
        "send_direct_message": "recipient",
        "send_channel_message": "channel",
        "post_webpage": "url",
        "invite_user_to_slack": "user_email",
        "add_user_to_channel": "channel",
        "remove_user_from_slack": "user",
    },
    "travel": {
        "send_email": "recipients",
        "reserve_hotel": "hotel",
        "reserve_restaurant": "restaurant",
        "reserve_car_rental": "company",
    },
    "workspace": {
        "send_email": "recipients",
        "share_file": "email",
    },
}


def _check_suite(suite_name: str) -> None:
    if suite_name not in _UNTRUSTED_READERS:
        known = ", ".join(AGENTDOJO_SUITES)
        raise ValueError(f"unknown AgentDojo suite {suite_name!r} (known suites: {known})")


def suite_untrusted_readers(suite_name: str) -> frozenset[str]:
    """Tools of ``suite_name`` whose returns can carry attacker-authored text."""
    _check_suite(suite_name)
    return _UNTRUSTED_READERS[suite_name]


def suite_external_sinks(suite_name: str) -> frozenset[str]:
    """Tools of ``suite_name`` with outward-visible or privileged effects."""
    _check_suite(suite_name)
    return _EXTERNAL_SINKS[suite_name]


def suite_taint_specs(suite_name: str) -> dict[str, ToolTaintSpec]:
    """Per-tool taint specs for ``suite_name``.

    Every untrusted reader adds :data:`AGENTDOJO_UNTRUSTED`; other tools get
    no spec (the gateway treats them as transparent propagators).
    """
    return {
        name: ToolTaintSpec.of(adds=(AGENTDOJO_UNTRUSTED,))
        for name in sorted(suite_untrusted_readers(suite_name))
    }


def suite_resource_args(suite_name: str) -> dict[str, str]:
    """Per-sink mapping of the argument carrying the call's destination.

    Feeds :class:`Selector` ``resource`` matching (e.g. an allow-list of
    recipients); the default ``policies/agentdojo.yaml`` does not use it,
    but episode audit records still capture the resource.
    """
    _check_suite(suite_name)
    return dict(_RESOURCE_ARGS[suite_name])


def gate_suite(
    gateway: Gateway,
    suite: Any,
    *,
    runtime: Any = None,
    agent_id: str | None = None,
) -> GatedAgentDojoRuntime:
    """Mount an AgentDojo task suite behind ``gateway``.

    ``suite`` is an ``agentdojo`` ``TaskSuite`` (its ``name`` selects the
    classification tables and its ``tools`` populate the runtime) or, when
    ``runtime`` is supplied, just the suite name. With ``runtime=None`` a
    fresh ``agentdojo.functions_runtime.FunctionsRuntime`` is built from
    ``suite.tools`` — the only code path that imports ``agentdojo``.

    Raises :class:`ValueError` for an unknown suite name, or when the
    runtime is missing tools the tables classify — the drift signal for a
    future benchmark version renaming its tools. Audit records are stamped
    ``agentdojo:<suite>`` unless ``agent_id`` overrides it.
    """
    suite_name = suite if isinstance(suite, str) else suite.name
    specs = suite_taint_specs(suite_name)

    if runtime is None:
        if isinstance(suite, str):
            raise TypeError(
                "gate_suite() needs a TaskSuite object to build a runtime; "
                "pass runtime=... when identifying the suite by name"
            )
        from agentdojo.functions_runtime import FunctionsRuntime

        runtime = FunctionsRuntime(suite.tools)

    registered = set(runtime.functions)
    classified = suite_untrusted_readers(suite_name) | suite_external_sinks(suite_name)
    missing = sorted(classified - registered)
    if missing:
        raise ValueError(
            f"suite {suite_name!r} runtime is missing classified tools: "
            f"{', '.join(missing)} (AgentDojo version drift? tables target "
            f"{AGENTDOJO_SUITE_VERSION})"
        )

    return wrap_agentdojo_runtime(
        gateway,
        runtime,
        taint_specs=specs,
        resource_args=suite_resource_args(suite_name),
        agent_id=agent_id or f"agentdojo:{suite_name}",
    )


__all__ = [
    "AGENTDOJO_SUITES",
    "AGENTDOJO_SUITE_VERSION",
    "AGENTDOJO_UNTRUSTED",
    "gate_suite",
    "suite_external_sinks",
    "suite_resource_args",
    "suite_taint_specs",
    "suite_untrusted_readers",
]
