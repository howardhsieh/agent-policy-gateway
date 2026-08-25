"""AgentDojo runtime adapter for agent-policy-gateway (R49a).

Mounts an `AgentDojo <https://github.com/ethz-spylab/agentdojo>`_
``FunctionsRuntime`` behind a :class:`Gateway` so every function call an
agent makes inside an AgentDojo episode is policy-mediated::

    from agent_policy_gateway import Gateway, wrap_agentdojo_runtime
    gateway = Gateway(policies=[...])
    gated = wrap_agentdojo_runtime(
        gateway,
        runtime,
        taint_specs={"read_inbox": ToolTaintSpec.of(adds=("agentdojo:untrusted",))},
        resource_args={"send_email": "recipient"},
    )
    result, error = gated.run_function(env, "send_email", {...})

This module deliberately does **not** import ``agentdojo``. The wrapper is
duck-typed against the ``FunctionsRuntime`` surface (verified against
agentdojo's real signatures, tested with hand-rolled fakes, mirroring the
LangChain/OpenAI/Anthropic/MCP adapters), so the package ships without the
benchmark installed. The surface relied upon:

* ``runtime.functions`` — a mapping of function name to descriptor.
* ``runtime.register_function(...)`` — optional; delegated when present.
* ``runtime.run_function(env, function, kwargs, raise_on_error=False)``
  returning ``(result, error | None)`` where an error is the string
  ``"ErrorType: message"`` — AgentDojo's native tool-error convention.

Two behaviors are specific to this adapter:

**Refusals are model-visible errors.** With ``raise_on_error=False`` (the
AgentDojo pipeline default) a policy refusal returns
``("", "PolicyDenied: …")`` / ``("", "PolicyReview: …")`` in the same
``"ErrorType: message"`` shape AgentDojo itself uses, so the episode
continues and the agent sees actionable feedback — exactly what the R50
benchmark needs to measure utility under defense. With
``raise_on_error=True`` the :class:`PolicyDenied` / :class:`PolicyReview`
exception propagates.

**Session-level taint accumulation.** AgentDojo's LLM loop cannot thread
``apg_input_label`` through its function calls, so the wrapper keeps a
cumulative :class:`TaintLabel` for the episode: each *executed* call's
``decision.output_label`` (which already joins the input label with the
tool's ``adds`` and strips its declassifies) becomes the input label of the
next call. Denied calls contribute nothing — their output never enters the
conversation. :attr:`GatedAgentDojoRuntime.taint_label` exposes the current
label and :meth:`GatedAgentDojoRuntime.reset_taint` resets it between
episodes.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from agent_policy_gateway.core import TaintLabel, ToolCall
from agent_policy_gateway.gateway import (
    Gateway,
    PolicyDenied,
    PolicyReview,
)
from agent_policy_gateway.taint import ToolTaintSpec

#: Default ``agent_id`` stamped on audit records when none is supplied.
DEFAULT_AGENTDOJO_AGENT_ID = "agentdojo"


def _format_error(exc: BaseException) -> str:
    """Render ``exc`` in AgentDojo's ``"ErrorType: message"`` convention."""
    return f"{type(exc).__name__}: {exc}"


def _format_refusal_error(exc: PolicyDenied | PolicyReview) -> str:
    """Render a policy refusal in the ``"ErrorType: message"`` convention.

    Unlike :func:`_format_error`, the rule id is spelled out so the R50
    benchmark can attribute refusals to rules straight from the episode
    transcript, and the model gets feedback it can act on.
    """
    decision = exc.decision
    rule = decision.rule_id or "(no rule - default disposition)"
    reason = f": {decision.reason}" if decision.reason else ""
    return f"{type(exc).__name__}: refused by rule '{rule}'{reason}"


class GatedAgentDojoRuntime:
    """A policy-mediated view over an AgentDojo ``FunctionsRuntime``.

    Duck-typed to the runtime surface AgentDojo's pipeline drives
    (``functions`` / ``register_function`` / ``run_function``), so it can
    stand in wherever the wrapped runtime was used. Construct via
    :func:`wrap_agentdojo_runtime`, which also registers the taint specs.
    """

    def __init__(
        self,
        gateway: Gateway,
        runtime: Any,
        *,
        resource_args: Mapping[str, str] | None = None,
        agent_id: str | None = None,
    ) -> None:
        if not callable(getattr(runtime, "run_function", None)):
            raise TypeError(
                "runtime object has no callable 'run_function': "
                f"{runtime!r}"
            )
        self._gateway = gateway
        self._runtime = runtime
        self._resource_args = dict(resource_args or {})
        self._agent_id = agent_id or DEFAULT_AGENTDOJO_AGENT_ID
        self._label = TaintLabel()

    # ----- duck-typed FunctionsRuntime surface --------------------------------

    @property
    def functions(self) -> Any:
        """The wrapped runtime's function registry, verbatim."""
        return self._runtime.functions

    def register_function(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate registration to the wrapped runtime."""
        return self._runtime.register_function(*args, **kwargs)

    def run_function(
        self,
        env: Any,
        function: str,
        kwargs: Mapping[str, Any],
        raise_on_error: bool = False,
    ) -> tuple[Any, str | None]:
        """Mediate one AgentDojo function call through the gateway.

        The call's input label is the episode's accumulated
        :attr:`taint_label`. On an executed call (ALLOW or REDACT) the
        decision's output label becomes the new accumulated label and the
        wrapped runtime's ``(result, error)`` tuple is returned — a tool
        error inside the runtime is rendered in AgentDojo's
        ``"ErrorType: message"`` convention (or re-raised when
        ``raise_on_error=True``), and an errored call does **not** update
        the accumulated label. On DENY/REVIEW the tool never runs, the
        label is unchanged, and the refusal is returned as
        ``("", "PolicyDenied: …")`` — or re-raised when
        ``raise_on_error=True``.
        """
        arguments = dict(kwargs)

        resource: str | None = None
        resource_arg_name = self._resource_args.get(function)
        if resource_arg_name is not None:
            value = arguments.get(resource_arg_name)
            if value is not None:
                resource = value if isinstance(value, str) else str(value)

        call = ToolCall(
            tool_name=function,
            args=arguments,
            input_label=self._label,
            agent_id=self._agent_id,
            call_id=uuid.uuid4().hex,
        )

        def _invoke(**tool_args: Any) -> tuple[Any, str | None]:
            # raise_on_error=True so a tool failure surfaces as the
            # original exception; the except clause below restores the
            # (result, error) contract for the raise_on_error=False path.
            # Keyword-arg binding (**tool_args) lets a REDACT rule mask
            # fields by argument name before the runtime sees them.
            return self._runtime.run_function(
                env, function, tool_args, raise_on_error=True
            )

        try:
            value, decision = self._gateway.execute(
                call, _invoke, resource=resource, **arguments
            )
        except (PolicyDenied, PolicyReview) as exc:
            if raise_on_error:
                raise
            return "", _format_refusal_error(exc)
        except Exception as exc:  # tool/runtime error, policy allowed it
            if raise_on_error:
                raise
            return "", _format_error(exc)

        self._label = decision.output_label
        return value

    def __getattr__(self, name: str) -> Any:
        """Fall back to the wrapped runtime for any other attribute.

        Keeps the wrapper drop-in for pipeline code touching runtime
        members beyond the three mediated above. Underscore-prefixed
        lookups are never delegated (guards recursion during
        construction/copying).
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._runtime, name)

    # ----- taint bookkeeping --------------------------------------------------

    @property
    def taint_label(self) -> TaintLabel:
        """The episode's accumulated taint label."""
        return self._label

    def reset_taint(self) -> None:
        """Reset the accumulated label (call between episodes)."""
        self._label = TaintLabel()


def wrap_agentdojo_runtime(
    gateway: Gateway,
    runtime: Any,
    *,
    taint_specs: Mapping[str, ToolTaintSpec] | None = None,
    resource_args: Mapping[str, str] | None = None,
    agent_id: str | None = None,
) -> GatedAgentDojoRuntime:
    """Mount an AgentDojo-style ``runtime`` behind ``gateway``.

    ``taint_specs`` maps function names to :class:`ToolTaintSpec`s and is
    registered with the gateway (e.g. untrusted-content readers ``add`` an
    ``agentdojo:untrusted`` source). ``resource_args`` declares per-function
    which argument carries the policy resource for :class:`Selector`
    ``resource`` matching. ``agent_id`` stamps audit records (default
    :data:`DEFAULT_AGENTDOJO_AGENT_ID`).
    """
    for name, spec in (taint_specs or {}).items():
        gateway.register_tool(name, spec)
    return GatedAgentDojoRuntime(
        gateway,
        runtime,
        resource_args=resource_args,
        agent_id=agent_id,
    )


__all__ = [
    "DEFAULT_AGENTDOJO_AGENT_ID",
    "GatedAgentDojoRuntime",
    "wrap_agentdojo_runtime",
]
