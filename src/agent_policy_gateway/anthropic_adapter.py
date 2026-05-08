"""Anthropic Messages-API tool-use adapter for agent-policy-gateway (R8).

This module exposes Python tools to an Anthropic Messages-API model in two
cooperating pieces:

1. :func:`anthropic_tool_specs` builds the JSON tool descriptors the model
   expects in the request payload (``tools=[...]``).
2. :func:`wrap_anthropic_tools` registers the same tools with a
   :class:`Gateway` and returns a ``{name: callable}`` map of
   gateway-mediated callables, which :func:`dispatch_anthropic_tool_use`
   then drives from the model's response.

The Anthropic Python SDK is **not** a dependency. ``tool_use`` blocks are
duck-typed against either the dict shape returned by raw HTTP responses
(``{"type": "tool_use", "id": ..., "name": ..., "input": {...}}``) or the
attribute shape produced by the SDK's pydantic models. Keeping the
dependency surface narrow lets the adapter be tested with hand-rolled
fakes and shipped to environments that have not adopted the
``anthropic`` SDK.

Differences from the OpenAI adapter (R7):

* The Messages tool spec is flat — ``{"name", "description", "input_schema"}``
  — with no ``"type": "function"`` / ``"function": {...}`` wrapper.
* Tool input arrives **already parsed** (``input`` is a JSON object), not
  a JSON-encoded string. We still tolerate a JSON string for robustness
  against intermediaries that re-serialised it.
* Results come back as ``tool_result`` *content blocks*, not chat
  messages — the caller wraps them in a ``{"role": "user", "content":
  [...]}`` turn alongside any other blocks they care about.
* Policy refusals set ``is_error: True`` on the result block; the model's
  next turn sees it that way and can react.

Design contract:

* ``AnthropicTool`` is a flat, frozen description: name, human prose,
  JSON Schema for ``input``, the Python function, plus the same
  ``resource_arg`` / ``taint_spec`` knobs the rest of the project uses.
* Reserved kwargs (``apg_input_label``, ``apg_agent_id``, ``apg_call_id``,
  ``apg_resource``) are honoured by the gateway-wrapped callable and are
  *not* surfaced in the JSON Schema sent to the model — they're
  orchestration concerns, not tool parameters. ``apg_call_id`` is
  auto-populated from the ``tool_use.id`` so audit records line up with
  the model's view.
* :func:`dispatch_anthropic_tool_use` always produces a ``tool_result``
  content block: on allow it carries the function result encoded as a
  JSON string, on policy denial / review it carries the JSON produced
  by ``on_denied`` (default: a structured ``{"error":
  "policy_refusal", ...}`` payload) with ``is_error: True``. Tool
  exceptions and malformed model output raise so the caller can
  decide.

Schema validation is deferred. The model usually produces conforming
JSON, but malformed values currently surface as a downstream
``TypeError`` inside the wrapped function rather than a structured
pre-call rejection.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_policy_gateway.core import TaintLabel
from agent_policy_gateway.gateway import (
    AGENT_ID_KWARG,
    CALL_ID_KWARG,
    INPUT_LABEL_KWARG,
    RESOURCE_KWARG,
    Gateway,
    GatewayError,
    PolicyDenied,
    PolicyReview,
)
from agent_policy_gateway.taint import ToolTaintSpec


@dataclass(frozen=True)
class AnthropicTool:
    """A Python tool exposed to an Anthropic Messages-API model.

    Attributes:
        name: Tool name. Anthropic constrains names to ``^[a-zA-Z0-9_-]{1,128}$``;
            we don't re-validate that pattern here, but empty strings are
            rejected upfront.
        description: Human-readable description shown to the model.
        input_schema: JSON Schema (``type: object``) describing the
            tool's ``input`` argument. Sent verbatim to the model.
        function: The Python callable to invoke when the model emits a
            ``tool_use`` block for this tool. Will be gateway-wrapped by
            :func:`wrap_anthropic_tools`.
        resource_arg: Optional argument name carrying the policy
            *resource* (matched against :class:`Selector` ``resource``).
            Works for arguments produced by the model in the parsed
            ``input`` payload.
        taint_spec: Optional :class:`ToolTaintSpec` registered with the
            gateway under this tool's (possibly prefixed) name.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    function: Callable[..., Any]
    resource_arg: str | None = None
    taint_spec: ToolTaintSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"AnthropicTool.name must be a non-empty string, got {self.name!r}"
            )
        if not callable(self.function):
            raise TypeError(
                f"AnthropicTool.function must be callable, got {self.function!r}"
            )
        if not isinstance(self.input_schema, Mapping):
            raise TypeError(
                "AnthropicTool.input_schema must be a JSON Schema mapping, got "
                f"{type(self.input_schema).__name__}"
            )


class AnthropicToolUseError(ValueError):
    """Raised when a model-produced ``tool_use`` block cannot be dispatched.

    Specifically: missing fields, malformed ``input``, or an unknown tool
    name. The message includes the offending payload to aid debugging —
    callers should not echo it verbatim to untrusted upstreams without
    a redaction step of their own.
    """


# --------------------------------------------------------------------------- #
# Spec generation                                                             #
# --------------------------------------------------------------------------- #


def _registered_name(tool: AnthropicTool, prefix: str | None) -> str:
    return f"{prefix}.{tool.name}" if prefix else tool.name


def anthropic_tool_specs(
    tools: Iterable[AnthropicTool],
    *,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Build the Anthropic ``tools=[...]`` array for an iterable of tools.

    Each entry has the shape::

        {
            "name": "<prefix.>name",
            "description": "...",
            "input_schema": {... JSON Schema ...},
        }

    The returned list is order-preserving and contains independent dicts
    (a recursive copy of ``input_schema``), so the caller can mutate
    either the spec list or the source tools without affecting the other.
    """
    seen: set[str] = set()
    specs: list[dict[str, Any]] = []
    for tool in tools:
        name = _registered_name(tool, prefix)
        if name in seen:
            raise ValueError(
                f"Anthropic tool spec collision for name {name!r} - "
                "tool names must be unique within a binding"
            )
        seen.add(name)
        specs.append(
            {
                "name": name,
                "description": tool.description,
                "input_schema": _copy_schema(tool.input_schema),
            }
        )
    return specs


def _copy_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive copy of a JSON Schema mapping (dicts and lists)."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if isinstance(value, Mapping):
            out[key] = _copy_schema(value)
        elif isinstance(value, list):
            out[key] = [_copy_schema(v) if isinstance(v, Mapping) else v for v in value]
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Registration                                                                #
# --------------------------------------------------------------------------- #


def wrap_anthropic_tools(
    gateway: Gateway,
    tools: Iterable[AnthropicTool],
    *,
    prefix: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """Mount each ``tool`` under ``gateway`` and return a name -> callable map.

    Each returned callable is the result of :meth:`Gateway.wrap_tool`, so
    invocations go through the policy + audit + taint pipeline. The
    callable accepts the tool's declared parameters as keyword arguments
    plus the four reserved gateway kwargs.

    ``prefix`` namespaces every tool's *registered* name (the name the
    gateway and audit log see, and the name emitted in
    :func:`anthropic_tool_specs`). The Python function itself is unchanged.
    """
    out: dict[str, Callable[..., Any]] = {}
    seen: set[str] = set()
    for tool in tools:
        registered = _registered_name(tool, prefix)
        if registered in seen:
            raise ValueError(
                f"Anthropic tool registration collision for name {registered!r}"
            )
        seen.add(registered)
        wrapped = gateway.wrap_tool(
            tool.function,
            tool_name=registered,
            taint_spec=tool.taint_spec,
            resource_arg=tool.resource_arg,
        )
        out[registered] = wrapped
    return out


# --------------------------------------------------------------------------- #
# Dispatch                                                                    #
# --------------------------------------------------------------------------- #


def _extract_tool_use_fields(tool_use: Any) -> tuple[str, str, Any]:
    """Pull ``(use_id, tool_name, raw_input)`` from either shape.

    Accepts the dict form
    ``{"type": "tool_use", "id": ..., "name": ..., "input": {...}}`` and
    the attribute form ``tool_use.id`` / ``tool_use.name`` /
    ``tool_use.input`` (as produced by the ``anthropic`` SDK's pydantic
    models). The ``type`` field, if present, must be ``"tool_use"``.
    """
    if isinstance(tool_use, Mapping):
        block_type = tool_use.get("type")
        use_id = tool_use.get("id")
        name = tool_use.get("name")
        raw_input = tool_use.get("input", {})
    else:
        block_type = getattr(tool_use, "type", None)
        use_id = getattr(tool_use, "id", None)
        name = getattr(tool_use, "name", None)
        raw_input = getattr(tool_use, "input", {})

    if block_type is not None and block_type != "tool_use":
        raise AnthropicToolUseError(
            f"expected a 'tool_use' content block, got type={block_type!r}"
        )
    if not isinstance(use_id, str) or not use_id:
        raise AnthropicToolUseError(
            f"tool_use id must be a non-empty string, got {use_id!r}"
        )
    if not isinstance(name, str) or not name:
        raise AnthropicToolUseError(
            f"tool_use name must be a non-empty string, got {name!r}"
        )
    return use_id, name, raw_input


def _coerce_input(name: str, raw: Any) -> dict[str, Any]:
    """Normalise ``tool_use.input`` to a plain ``dict``.

    Anthropic models emit ``input`` as a JSON object (parsed), but we
    tolerate a JSON-encoded string for robustness against intermediaries
    that re-serialised it. ``None`` and empty-string both decode to
    ``{}``.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnthropicToolUseError(
                f"tool_use {name!r}: input is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed, dict):
            raise AnthropicToolUseError(
                f"tool_use {name!r}: input must decode to a JSON object, "
                f"got {type(parsed).__name__}"
            )
        return parsed
    raise AnthropicToolUseError(
        f"tool_use {name!r}: input must be a mapping or JSON string, got "
        f"{type(raw).__name__}"
    )


def _default_on_denied(error: GatewayError, tool_use: Mapping[str, Any]) -> str:
    """Default ``tool_result`` body for a policy refusal.

    Encoded as a JSON string so the model sees structured, easily-parsed
    feedback. ``tool_use`` is accepted only to match the ``on_denied``
    signature; we don't echo any of its fields here to keep the surface
    predictable.
    """
    del tool_use  # signature parity only
    decision = error.decision
    return json.dumps(
        {
            "error": "policy_refusal",
            "verdict": decision.verdict.value,
            "rule_id": decision.rule_id,
            "reason": decision.reason or "",
        }
    )


def _result_to_content(result: Any) -> str:
    """Render a tool result as the ``content`` string of a tool_result block."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return json.dumps({"value": str(result)})


def dispatch_anthropic_tool_use(
    wrapped: Mapping[str, Callable[..., Any]],
    tool_use: Any,
    *,
    input_label: TaintLabel | None = None,
    agent_id: str | None = None,
    resource: str | None = None,
    on_denied: Callable[[GatewayError, Mapping[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Dispatch one model-produced ``tool_use`` block through the gateway.

    Returns a ``tool_result`` content block ready to be appended to the
    ``content`` array of a ``role="user"`` turn. On allow the block
    contains the JSON-encoded tool result; on policy denial / review it
    carries the JSON produced by ``on_denied`` (default: a structured
    ``{"error": "policy_refusal", ...}`` payload) with
    ``is_error: True``.

    Raises :class:`AnthropicToolUseError` for malformed input or unknown
    tool names. Re-raises any exception thrown by the underlying tool —
    those represent a tool-side bug or environmental failure that the
    surrounding orchestration loop should handle, not something the
    model should be invited to retry against.
    """
    use_id, name, raw_input = _extract_tool_use_fields(tool_use)
    if name not in wrapped:
        raise AnthropicToolUseError(
            f"unknown tool name {name!r}; "
            f"expected one of: {sorted(wrapped)!r}"
        )
    parsed = _coerce_input(name, raw_input)

    forwarded: dict[str, Any] = dict(parsed)
    if input_label is not None:
        forwarded[INPUT_LABEL_KWARG] = input_label
    if agent_id is not None:
        forwarded[AGENT_ID_KWARG] = agent_id
    if resource is not None:
        forwarded[RESOURCE_KWARG] = resource
    forwarded[CALL_ID_KWARG] = use_id

    try:
        result = wrapped[name](**forwarded)
    except (PolicyDenied, PolicyReview) as exc:
        body = (on_denied or _default_on_denied)(
            exc,
            tool_use if isinstance(tool_use, Mapping) else {"id": use_id, "name": name},
        )
        return {
            "type": "tool_result",
            "tool_use_id": use_id,
            "content": body,
            "is_error": True,
        }

    return {
        "type": "tool_result",
        "tool_use_id": use_id,
        "content": _result_to_content(result),
    }


def dispatch_anthropic_tool_uses(
    wrapped: Mapping[str, Callable[..., Any]],
    tool_uses: Iterable[Any],
    *,
    input_label: TaintLabel | None = None,
    agent_id: str | None = None,
    resource: str | None = None,
    on_denied: Callable[[GatewayError, Mapping[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Dispatch each ``tool_use`` in ``tool_uses`` and collect the result blocks.

    Convenience wrapper around :func:`dispatch_anthropic_tool_use`; the
    same ``input_label`` / ``agent_id`` / ``resource`` apply to every
    use in the batch. Callers that need to vary these per-use should
    iterate manually.
    """
    return [
        dispatch_anthropic_tool_use(
            wrapped,
            tu,
            input_label=input_label,
            agent_id=agent_id,
            resource=resource,
            on_denied=on_denied,
        )
        for tu in tool_uses
    ]


__all__ = [
    "AnthropicTool",
    "AnthropicToolUseError",
    "anthropic_tool_specs",
    "dispatch_anthropic_tool_use",
    "dispatch_anthropic_tool_uses",
    "wrap_anthropic_tools",
]
