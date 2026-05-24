"""Reference monitor: Gateway class and wrap_tool decorator (R4 + R9).

This module wires the static pieces of the project — the core types in
:mod:`agent_policy_gateway.core`, the taint algebra in
:mod:`agent_policy_gateway.taint`, and the policy DSL in
:mod:`agent_policy_gateway.policy` — into a runtime *reference monitor*.

A :class:`Gateway` holds an ordered list of :class:`Policy` objects, a
registry mapping tool names to :class:`ToolTaintSpec`, and an optional
audit-log writer. Every call goes through one of four entry points:

* :meth:`Gateway.execute` — the synchronous workhorse. Takes a
  fully-built :class:`ToolCall`, walks policies in order, applies the
  first matching rule (or the default), propagates taint to compute the
  output label, writes an audit record, and either invokes the
  underlying function or raises.
* :meth:`Gateway.aexecute` — the asynchronous twin. Same contract as
  :meth:`execute`, but ``fn`` may be a coroutine function (or any
  callable returning an awaitable) and the audit writer may itself
  return an awaitable, in which case it is awaited before the tool
  runs. Sync audit writers continue to work unchanged.
* :meth:`Gateway.wrap_tool` — sync sugar over ``execute``. Wraps a
  plain Python function so each call is mediated by the gateway.
* :meth:`Gateway.wrap_tool_async` — async sugar over ``aexecute``.
  Wraps an ``async def`` function so each ``await tool(...)`` is
  mediated by the gateway.

In every entry point the same four reserved keyword arguments
(``apg_input_label``, ``apg_agent_id``, ``apg_call_id``,
``apg_resource``) configure the call and are stripped before the
wrapped function sees them.

Audit-log writing uses any ``Callable[[ToolCall, Decision], None]``;
:class:`agent_policy_gateway.audit.JsonlAuditWriter` is the on-disk
implementation, paired with the ``apg-replay`` CLI for reading logs
back as a human-readable timeline. Async callers may pass a writer
whose return value is awaitable — :meth:`aexecute` awaits it as part
of the fail-closed-on-audit contract.

Rate limiting is enforced (R16). The policy DSL allows ``action:
rate_limit`` with a ``limit_per_minute`` field; the gateway pairs each
matching rule with a :class:`~agent_policy_gateway.ratelimit.RateLimiter`
— a sliding-window counter keyed by ``(agent_id, tool_name)``. The pure
:meth:`Gateway.decide` *peeks* the limiter and reports ALLOW while the
window has room or DENY once it is full; the stateful entry points
(:meth:`Gateway.execute` / :meth:`Gateway.aexecute`) *consume* a slot so
that only calls that actually run count against the budget. The N+1th
call inside the window is refused and audited like any other denial.
"""

from __future__ import annotations

import functools
import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from agent_policy_gateway.core import Decision, TaintLabel, ToolCall, Verdict
from agent_policy_gateway.policy import Action, Policy, RedactSpec, Rule
from agent_policy_gateway.ratelimit import RateLimiter
from agent_policy_gateway.taint import ToolTaintSpec, propagate

# Reserved kwargs the wrapper recognises and strips before forwarding to
# the wrapped function. Surfaced as module constants so tests and callers
# do not have to spell them as raw strings.
INPUT_LABEL_KWARG = "apg_input_label"
AGENT_ID_KWARG = "apg_agent_id"
CALL_ID_KWARG = "apg_call_id"
RESOURCE_KWARG = "apg_resource"

_RESERVED_KWARGS: frozenset[str] = frozenset(
    {INPUT_LABEL_KWARG, AGENT_ID_KWARG, CALL_ID_KWARG, RESOURCE_KWARG}
)

AuditWriter = Callable[[ToolCall, Decision], None]


class GatewayError(Exception):
    """Base class for exceptions raised by the gateway.

    Carries both the :class:`Decision` and the :class:`ToolCall` so
    callers that catch a refusal can audit *why* without reaching back
    into the gateway.
    """

    def __init__(self, message: str, *, decision: Decision, call: ToolCall) -> None:
        super().__init__(message)
        self.decision = decision
        self.call = call


class PolicyDenied(GatewayError):
    """Raised when a policy denies a tool call."""


class PolicyReview(GatewayError):
    """Raised when a policy requires human review.

    Until a reviewer is wired in (later milestone) the gateway treats
    REVIEW as a hard refusal at runtime. The :class:`Decision` is still
    available on the exception for callers that want to defer the call.
    """


def _action_to_verdict(action: Action) -> Verdict:
    """Map a policy :class:`Action` to a runtime :class:`Verdict`.

    ``rate_limit`` maps to ``ALLOW`` here as a fallback; the gateway
    resolves the real allow/deny against its :class:`RateLimiter` inside
    :meth:`Gateway._decide`. This branch is only reached for a
    rate-limit rule if no limiter is configured, which the
    default-constructed gateway never does.
    """
    if action == Action.ALLOW:
        return Verdict.ALLOW
    if action == Action.DENY:
        return Verdict.DENY
    if action == Action.REVIEW:
        return Verdict.REVIEW
    if action == Action.RATE_LIMIT:
        return Verdict.ALLOW
    if action == Action.REDACT:
        return Verdict.REDACT
    raise ValueError(f"unknown policy action: {action!r}")  # pragma: no cover


@dataclass
class Gateway:
    """Reference monitor for AI agent tool calls.

    Attributes:
        policies: Ordered list of :class:`Policy` objects. Within a
            policy, rules are matched in order; across policies, the
            first policy with a match wins.
        tool_specs: Map from tool name to its declared
            :class:`ToolTaintSpec`. Tools without a spec are treated as
            transparent propagators (no intrinsic sources, no
            declassification).
        audit_writer: Optional callable invoked exactly once per
            decision with the :class:`ToolCall` and resulting
            :class:`Decision`. Invoked *before* the underlying
            function runs, so a failed audit blocks the call.
        default_deny: When True, calls that match no rule are denied.
            Default False (a permissive baseline; tighten with explicit
            policies in production).
        rate_limiter: Sliding-window counter backing ``rate_limit`` rules,
            keyed by ``(agent_id, tool_name)``. A fresh
            :class:`RateLimiter` (60s window, monotonic clock) by default;
            inject one with a custom window or clock for testing.
    """

    policies: list[Policy] = field(default_factory=list)
    tool_specs: dict[str, ToolTaintSpec] = field(default_factory=dict)
    audit_writer: AuditWriter | None = None
    default_deny: bool = False
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)

    # ----- registration helpers -------------------------------------------------

    def add_policy(self, policy: Policy) -> None:
        """Append a :class:`Policy` to the gateway."""
        self.policies.append(policy)

    def register_tool(self, tool_name: str, spec: ToolTaintSpec) -> None:
        """Associate a :class:`ToolTaintSpec` with a tool name.

        Re-registering the same name overwrites the previous spec.
        """
        self.tool_specs[tool_name] = spec

    # ----- core execution -------------------------------------------------------

    def decide(self, call: ToolCall, *, resource: str | None = None) -> Decision:
        """Evaluate policies for ``call`` and return a :class:`Decision`.

        Pure: does not invoke the underlying tool, does not write to the
        audit log, does not raise on DENY/REVIEW. Useful for tests and
        for callers that want to inspect a verdict ahead of time.

        For a ``rate_limit`` rule this *peeks* the :class:`RateLimiter`
        (read-only): it reports ALLOW while the window has room and DENY
        once it is full, without consuming a slot. The slot is only
        consumed when the call actually runs through :meth:`execute` or
        :meth:`aexecute`.
        """
        return self._decide(call, resource=resource, consume=False)

    def _match_rule(
        self, call: ToolCall, *, resource: str | None = None
    ) -> tuple[Rule | None, TaintLabel]:
        """Return the first matching rule (or ``None``) and the output label.

        Across policies the first policy with a match wins; within a
        policy the first matching rule wins (delegated to
        :meth:`Policy.first_match`).
        """
        spec = self.tool_specs.get(call.tool_name)
        output_label = propagate([call.input_label], spec)
        for policy in self.policies:
            rule = policy.first_match(call, resource=resource)
            if rule is not None:
                return rule, output_label
        return None, output_label

    def _decide(
        self,
        call: ToolCall,
        *,
        resource: str | None = None,
        consume: bool = False,
    ) -> Decision:
        """Shared decision logic for :meth:`decide` and the execute paths.

        When ``consume`` is False the rate-limit check is read-only (a
        dry run, used by :meth:`decide`). When True a slot is consumed
        for an allowed ``rate_limit`` call (used by :meth:`execute` /
        :meth:`aexecute`); an over-limit call consumes nothing and is
        returned as a DENY.
        """
        rule, output_label = self._match_rule(call, resource=resource)
        if rule is None:
            if self.default_deny:
                return Decision(
                    verdict=Verdict.DENY,
                    rule_id=None,
                    reason="default-deny: no policy rule matched",
                    output_label=output_label,
                )
            return Decision(
                verdict=Verdict.ALLOW,
                rule_id=None,
                reason="default-allow: no policy rule matched",
                output_label=output_label,
            )
        if rule.effect.action == Action.REDACT and rule.effect.redact is not None:
            spec = rule.effect.redact
            declassified = (
                output_label.sources - set(spec.declassify)
            ) | set(spec.add_label)
            return Decision(
                verdict=Verdict.REDACT,
                rule_id=rule.id,
                reason=rule.effect.reason,
                output_label=TaintLabel(frozenset(declassified)),
            )
        if (
            rule.effect.action == Action.RATE_LIMIT
            and self.rate_limiter is not None
            and rule.effect.limit_per_minute is not None
        ):
            key = (call.agent_id, call.tool_name)
            limit = rule.effect.limit_per_minute
            if consume:
                allowed = self.rate_limiter.try_acquire(key, limit)
            else:
                allowed = self.rate_limiter.peek(key, limit)
            if allowed:
                return Decision(
                    verdict=Verdict.ALLOW,
                    rule_id=rule.id,
                    reason=rule.effect.reason,
                    output_label=output_label,
                )
            reason = (
                f"rate limit exceeded: more than {limit} calls/min "
                f"for tool {call.tool_name!r}"
            )
            if rule.effect.reason:
                reason = f"{reason} ({rule.effect.reason})"
            return Decision(
                verdict=Verdict.DENY,
                rule_id=rule.id,
                reason=reason,
                output_label=output_label,
            )
        return Decision(
            verdict=_action_to_verdict(rule.effect.action),
            rule_id=rule.id,
            reason=rule.effect.reason,
            output_label=output_label,
        )

    def _redact_spec_for(
        self, call: ToolCall, *, resource: str | None = None
    ) -> RedactSpec | None:
        """Return the :class:`RedactSpec` of the rule matching ``call``.

        ``None`` when no rule matches or the matching rule is not a redact
        rule. Used by the execute paths to fetch the transformation to apply
        after :meth:`_decide` has already classified the call as REDACT.
        """
        rule, _ = self._match_rule(call, resource=resource)
        if rule is not None and rule.effect.action == Action.REDACT:
            return rule.effect.redact
        return None

    def _apply_redaction(
        self,
        call: ToolCall,
        decision: Decision,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        resource: str | None = None,
    ) -> tuple[ToolCall, Decision, tuple[Any, ...], dict[str, Any]]:
        """Mask the matched fields and record that redaction happened.

        Returns an updated ``(call, decision, fn_args, fn_kwargs)`` tuple:

        * ``call`` has its audit-visible ``args`` masked so the log never
          captures the raw sensitive value.
        * ``decision`` carries ``redacted_fields`` naming what was masked.
        * ``fn_args`` / ``fn_kwargs`` are the arguments the downstream tool
          will actually receive, with the matched fields masked in place.
        """
        spec = self._redact_spec_for(call, resource=resource)
        if spec is None:
            return call, decision, args, kwargs

        masked_args = dict(call.args)
        audit_changed: list[str] = []
        for fieldname in spec.fields:
            if fieldname in masked_args:
                new = spec.redact_value(masked_args[fieldname])
                if new != masked_args[fieldname]:
                    masked_args[fieldname] = new
                    audit_changed.append(fieldname)

        fn_args, fn_kwargs, fn_changed = _redact_call_arguments(spec, fn, args, kwargs)
        changed = list(dict.fromkeys(fn_changed + audit_changed))
        new_call = replace(call, args=masked_args)
        new_decision = replace(decision, redacted_fields=tuple(changed))
        return new_call, new_decision, fn_args, fn_kwargs

    def execute(
        self,
        call: ToolCall,
        fn: Callable[..., Any],
        *args: Any,
        resource: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Decision]:
        """Mediate a tool call through the gateway.

        Steps, in order:

        1. Build the :class:`Decision` from the policies and taint spec.
        2. Hand the ``(call, decision)`` pair to ``audit_writer`` if set.
           A raising audit writer aborts the call before the tool runs.
        3. If the verdict is :attr:`Verdict.ALLOW`, invoke
           ``fn(*args, **kwargs)`` and return ``(result, decision)``.
        4. If the verdict is :attr:`Verdict.DENY`, raise
           :class:`PolicyDenied`.
        5. If the verdict is :attr:`Verdict.REVIEW`, raise
           :class:`PolicyReview`.
        """
        decision = self._decide(call, resource=resource, consume=True)
        fn_args, fn_kwargs = args, kwargs
        if decision.verdict == Verdict.REDACT:
            call, decision, fn_args, fn_kwargs = self._apply_redaction(
                call, decision, fn, args, kwargs, resource=resource
            )
        if self.audit_writer is not None:
            self.audit_writer(call, decision)
        if decision.verdict == Verdict.DENY:
            raise PolicyDenied(
                _format_refusal("deny", decision),
                decision=decision,
                call=call,
            )
        if decision.verdict == Verdict.REVIEW:
            raise PolicyReview(
                _format_refusal("review", decision),
                decision=decision,
                call=call,
            )
        result = fn(*fn_args, **fn_kwargs)
        return result, decision

    # ----- async core execution -------------------------------------------------

    async def aexecute(
        self,
        call: ToolCall,
        fn: Callable[..., Any],
        *args: Any,
        resource: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Decision]:
        """Asynchronous twin of :meth:`execute`.

        Same fail-closed-on-audit / DENY-or-REVIEW-raises contract as
        :meth:`execute`, with two extra affordances for async callers:

        * ``fn`` may be a coroutine function or any callable returning an
          awaitable. The result is awaited before being returned.
        * The audit writer's return value is awaited if it is itself an
          awaitable, so callers can plug in async log sinks (e.g. async
          databases, queues) without a thread shim. Sync writers — the
          baseline — keep working unchanged.

        Synchronous ``fn`` callables are still accepted for ergonomics
        (mixing one async tool with a few sync ones), but the recommended
        async use is :meth:`wrap_tool_async`, which always awaits.
        """
        decision = self._decide(call, resource=resource, consume=True)
        fn_args, fn_kwargs = args, kwargs
        if decision.verdict == Verdict.REDACT:
            call, decision, fn_args, fn_kwargs = self._apply_redaction(
                call, decision, fn, args, kwargs, resource=resource
            )
        if self.audit_writer is not None:
            audit_result = self.audit_writer(call, decision)
            if inspect.isawaitable(audit_result):
                await audit_result
        if decision.verdict == Verdict.DENY:
            raise PolicyDenied(
                _format_refusal("deny", decision),
                decision=decision,
                call=call,
            )
        if decision.verdict == Verdict.REVIEW:
            raise PolicyReview(
                _format_refusal("review", decision),
                decision=decision,
                call=call,
            )
        result = fn(*fn_args, **fn_kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result, decision

    # ----- decorator ------------------------------------------------------------

    def wrap_tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        tool_name: str | None = None,
        taint_spec: ToolTaintSpec | None = None,
        resource_arg: str | None = None,
    ) -> Callable[..., Any]:
        """Wrap a tool function so calls are mediated by this gateway.

        Usage as a parameterised decorator::

            gw = Gateway(policies=[load_policy("policies/default.yaml")])

            @gw.wrap_tool(
                tool_name="send_email",
                taint_spec=ToolTaintSpec.of(),
                resource_arg="to",
            )
            def send_email(to: str, body: str) -> dict:
                ...

            send_email(
                "ops@example.com",
                "hi",
                apg_input_label=TaintLabel.of("web"),
                apg_agent_id="agent.research",
            )

        Or as a bare decorator (``@gw.wrap_tool``) when defaults suffice.

        Reserved keyword arguments (stripped before the wrapped function
        sees them):

        * ``apg_input_label``: input :class:`TaintLabel`. Default: empty.
        * ``apg_agent_id``: identity string for selector matching.
        * ``apg_call_id``: caller-supplied id. Default: a fresh uuid4 hex.
        * ``apg_resource``: target resource for ``Selector.resource`` matching.
          Overrides ``resource_arg`` when both are provided.

        ``resource_arg`` names a *real* parameter of ``fn`` whose bound value
        should be matched against ``Selector.resource``. The wrapper binds
        ``fn``'s signature and looks up that parameter — works for both
        positional and keyword forms.

        ``taint_spec`` is registered with the gateway under ``tool_name`` (or
        ``fn.__name__`` if not given) so other code paths into the gateway
        — including direct :meth:`execute` calls — see the same behaviour.
        """

        def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            name = tool_name or target.__name__
            if taint_spec is not None:
                self.register_tool(name, taint_spec)
            sig = inspect.signature(target)

            @functools.wraps(target)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                input_label = kwargs.pop(INPUT_LABEL_KWARG, None) or TaintLabel()
                agent_id = kwargs.pop(AGENT_ID_KWARG, None)
                call_id = kwargs.pop(CALL_ID_KWARG, None) or uuid.uuid4().hex
                explicit_resource = kwargs.pop(RESOURCE_KWARG, None)

                # Bind through the function's signature so positional args
                # are recorded in the audit log alongside kwargs. Fall back
                # to the raw kwargs if the user passes an arity that does
                # not match the wrapped function — the call will fail
                # naturally when the function is invoked, but we still want
                # to log a sensible record before that happens.
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    call_args: dict[str, Any] = dict(bound.arguments)
                except TypeError:
                    call_args = dict(kwargs)

                # Strip any reserved keys that may have leaked through (eg.
                # if the wrapped function declared **kwargs).
                for k in _RESERVED_KWARGS:
                    call_args.pop(k, None)

                resource = explicit_resource
                if resource is None and resource_arg is not None:
                    resource = call_args.get(resource_arg)
                    if resource is not None and not isinstance(resource, str):
                        resource = str(resource)

                call = ToolCall(
                    tool_name=name,
                    args=call_args,
                    input_label=input_label,
                    agent_id=agent_id,
                    call_id=call_id,
                )
                value, _ = self.execute(
                    call, target, *args, resource=resource, **kwargs
                )
                return value

            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator

    # ----- async decorator ------------------------------------------------------

    def wrap_tool_async(
        self,
        fn: Callable[..., Awaitable[Any]] | None = None,
        *,
        tool_name: str | None = None,
        taint_spec: ToolTaintSpec | None = None,
        resource_arg: str | None = None,
    ) -> Callable[..., Any]:
        """Async sibling of :meth:`wrap_tool`.

        Wraps an ``async def`` function — or any callable that returns an
        awaitable — so each ``await tool(...)`` is mediated by this
        gateway. Reserved keyword arguments and ``resource_arg`` semantics
        match the synchronous wrapper exactly; the only difference is
        that the returned wrapper is itself an ``async def`` and routes
        through :meth:`aexecute`.

        Usage::

            gw = Gateway(policies=[load_policy("policies/default.yaml")])

            @gw.wrap_tool_async(
                tool_name="send_email",
                taint_spec=ToolTaintSpec.of(),
                resource_arg="to",
            )
            async def send_email(to: str, body: str) -> dict:
                ...

            await send_email(
                "ops@example.com",
                "hi",
                apg_input_label=TaintLabel.of("web"),
                apg_agent_id="agent.research",
            )
        """

        def decorator(target: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
            name = tool_name or target.__name__
            if taint_spec is not None:
                self.register_tool(name, taint_spec)
            sig = inspect.signature(target)

            @functools.wraps(target)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                input_label = kwargs.pop(INPUT_LABEL_KWARG, None) or TaintLabel()
                agent_id = kwargs.pop(AGENT_ID_KWARG, None)
                call_id = kwargs.pop(CALL_ID_KWARG, None) or uuid.uuid4().hex
                explicit_resource = kwargs.pop(RESOURCE_KWARG, None)

                # Same signature-binding strategy as wrap_tool: fall back
                # to raw kwargs if the caller's arity does not match the
                # wrapped function — the call will fail naturally when
                # invoked, but we still want a sensible audit record.
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    call_args: dict[str, Any] = dict(bound.arguments)
                except TypeError:
                    call_args = dict(kwargs)

                for k in _RESERVED_KWARGS:
                    call_args.pop(k, None)

                resource = explicit_resource
                if resource is None and resource_arg is not None:
                    resource = call_args.get(resource_arg)
                    if resource is not None and not isinstance(resource, str):
                        resource = str(resource)

                call = ToolCall(
                    tool_name=name,
                    args=call_args,
                    input_label=input_label,
                    agent_id=agent_id,
                    call_id=call_id,
                )
                value, _ = await self.aexecute(
                    call, target, *args, resource=resource, **kwargs
                )
                return value

            return wrapper

        if fn is not None:
            return decorator(fn)
        return decorator


def _redact_call_arguments(
    spec: RedactSpec,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], list[str]]:
    """Mask ``spec.fields`` in the arguments destined for ``fn``.

    Binds ``fn``'s signature so a field passed positionally is masked just
    like a keyword one, then rebuilds the call. If the signature cannot be
    introspected or bound (builtins, mismatched arity) the helper falls
    back to masking matching keyword arguments only — the call still runs
    and is logged, it just cannot reach positional-only fields.

    Returns ``(fn_args, fn_kwargs, changed_fields)``.
    """
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
    except (TypeError, ValueError):
        new_kwargs = dict(kwargs)
        changed: list[str] = []
        for fieldname in spec.fields:
            if fieldname in new_kwargs:
                new = spec.redact_value(new_kwargs[fieldname])
                if new != new_kwargs[fieldname]:
                    new_kwargs[fieldname] = new
                    changed.append(fieldname)
        return args, new_kwargs, changed

    changed = []
    for fieldname in spec.fields:
        if fieldname in bound.arguments:
            new = spec.redact_value(bound.arguments[fieldname])
            if new != bound.arguments[fieldname]:
                bound.arguments[fieldname] = new
                changed.append(fieldname)
    return bound.args, bound.kwargs, changed


def _format_refusal(prefix: str, decision: Decision) -> str:
    """Render a short human message for a refusal exception."""
    detail = decision.reason or decision.rule_id or "no reason"
    return f"{prefix}: {detail}"


__all__ = [
    "AGENT_ID_KWARG",
    "AuditWriter",
    "CALL_ID_KWARG",
    "Gateway",
    "GatewayError",
    "INPUT_LABEL_KWARG",
    "PolicyDenied",
    "PolicyReview",
    "RESOURCE_KWARG",
]
