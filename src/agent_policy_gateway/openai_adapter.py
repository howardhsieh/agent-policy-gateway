"""OpenAI function-calling adapter for agent-policy-gateway (R7).

This module exposes Python tools to an OpenAI-compatible Chat Completions /
Responses model in two cooperating pieces:

1. :func:`openai_tool_specs` builds the JSON tool descriptors the model
   expects in the request payload (``tools=[...]``).
2. :func:`wrap_openai_tools` registers the same tools with a
   :class:`Gateway` and returns a ``{name: callable}`` map of
   gateway-mediated callables, which :func:`dispatch_openai_tool_call`
   then drives from the model's response.

The OpenAI Python SDK is **not** a dependency — `tool_calls` are
duck-typed against either the dict shape returned by raw HTTP responses
(``{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json>"}}``)
or the attribute shape produced by the SDK's pydantic models. Keeping
the dependency surface narrow lets the adapter be tested with
hand-rolled fakes and shipped to environments that have not adopted the
``openai`` SDK.

Design contract:

* ``OpenAITool`` is a flat, frozen description: name, human prose, JSON
  Schema for arguments, the Python function, plus the same
  ``resource_arg`` / ``taint_spec`` knobs the rest of the project uses.
* Reserved kwargs (``apg_input_label``, ``apg_agent_id``, ``apg_call_id``,
  ``apg_resource``) are honoured by the gateway-wrapped callable and are
  *not* surfaced in the JSON Schema sent to the model — they're orchestration
  concerns, not function parameters.
* :func:`dispatch_openai_tool_call` always produces a ``role="tool"``
  message: on allow it carries the function result encoded as JSON, on
  policy denial / review it carries a structured error payload so the
  model can recover. Tool exceptions and malformed model output raise so
  the caller can decide.

Schema validation is deferred. The model usually produces conforming
JSON, but malformed values currently surface as a downstream ``TypeError``
inside the wrapped function rather than a structured pre-call rejection.
A future milestone may add jsonschema validation here.
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
class OpenAITool:
    """A Python tool exposed to an OpenAI function-calling model.

    Attributes:
        name: Tool name. Must match ``[a-zA-Z0-9_-]+``-ish (OpenAI's
            constraint); we don't re-validate that here, but empty
            strings are rejected upfront.
        description: Human-readable description shown to the model.
        parameters: JSON Schema (``type: object``) describing the
            tool's arguments. Sent verbatim to the model.
        function: The Python callable to invoke when the model calls
            this tool. Will be gateway-wrapped by
            :func:`wrap_openai_tools`.
        resource_arg: Optional argument name carrying the policy
            *resource* (matched against :class:`Selector` ``resource``).
            Works for arguments produced by the model in the parsed
            JSON ``arguments`` payload.
        taint_spec: Optional :class:`ToolTaintSpec` registered with the
            gateway under this tool's (possibly prefixed) name.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    function: Callable[..., Any]
    resource_arg: str | None = None
    taint_spec: ToolTaintSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"OpenAITool.name must be a non-empty string, got {self.name!r}"
            )
        if not callable(self.function):
            raise TypeError(
                f"OpenAITool.function must be callable, got {self.function!r}"
            )
        if not isinstance(self.parameters, Mapping):
            raise TypeError(
                "OpenAITool.parameters must be a JSON Schema mapping, got "
                f"{type(self.parameters).__name__}"
            )


class OpenAIToolCallError(ValueError):
    """Raised when a model-produced tool_call cannot be dispatched.

    Specifically: missing fields, unparseable JSON arguments, or an
    unknown tool name. The message includes the offending payload to
    aid debugging — callers should not echo it verbatim to untrusted
    upstreams without a redaction step of their own.
    """


# --------------------------------------------------------------------------- #
# Spec generation                                                             #
# --------------------------------------------------------------------------- #


def _registered_name(tool: OpenAITool, prefix: str | None) -> str:
    return f"{prefix}.{tool.name}" if prefix else tool.name


def openai_tool_specs(
    tools: Iterable[OpenAITool],
    *,
    prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Build the OpenAI ``tools=[...]`` array for an iterable of tools.

    Each entry has the shape::

        {
            "type": "function",
            "function": {
                "name": "<prefix.>name",
                "description": "...",
                "parameters": {... JSON Schema ...},
            },
        }

    The returned list is order-preserving and contains independent dicts
    (a recursive copy of ``parameters``), so the caller can mutate either
    the spec list or the source tools without affecting the other.
    """
    seen: set[str] = set()
    specs: list[dict[str, Any]] = []
    for tool in tools:
        name = _registered_name(tool, prefix)
        if name in seen:
            raise ValueError(
                f"OpenAI tool spec collision for name {name!r} - "
                "tool names must be unique within a binding"
            )
        seen.add(name)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": _copy_schema(tool.parameters),
                },
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


def wrap_openai_tools(
    gateway: Gateway,
    tools: Iterable[OpenAITool],
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
    :func:`openai_tool_specs`). The Python function itself is unchanged.
    """
    out: dict[str, Callable[..., Any]] = {}
    seen: set[str] = set()
    for tool in tools:
        registered = _registered_name(tool, prefix)
        if registered in seen:
            raise ValueError(
                f"OpenAI tool registration collision for name {registered!r}"
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


def _extract_tool_call_fields(tool_call: Any) -> tuple[str, str, str]:
    """Pull ``(call_id, function_name, arguments_str)`` from either shape.

    Accepts the dict form ``{"id": ..., "function": {"name": ..., "arguments": ...}}``
    and the attribute form ``tool_call.id`` / ``tool_call.function.name`` /
    ``tool_call.function.arguments``.
    """
    if isinstance(tool_call, Mapping):
        call_id = tool_call.get("id")
        fn = tool_call.get("function")
        if not isinstance(fn, Mapping):
            raise OpenAIToolCallError(
                f"tool_call missing 'function' object: {tool_call!r}"
            )
        name = fn.get("name")
        arguments = fn.get("arguments")
    else:
        call_id = getattr(tool_call, "id", None)
        fn = getattr(tool_call, "function", None)
        if fn is None:
            raise OpenAIToolCallError(
                f"tool_call missing 'function' attribute: {tool_call!r}"
            )
        name = getattr(fn, "name", None)
        arguments = getattr(fn, "arguments", None)

    if not isinstance(name, str) or not name:
        raise OpenAIToolCallError(
            f"tool_call function.name must be a non-empty string, got {name!r}"
        )
    if not isinstance(call_id, str) or not call_id:
        raise OpenAIToolCallError(
            f"tool_call id must be a non-empty string, got {call_id!r}"
        )
    if arguments is None:
        arguments = "{}"
    if not isinstance(arguments, str):
        raise OpenAIToolCallError(
            "tool_call function.arguments must be a JSON string, got "
            f"{type(arguments).__name__}"
        )
    return call_id, name, arguments


def _parse_arguments(name: str, arguments: str) -> dict[str, Any]:
    if arguments == "":
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise OpenAIToolCallError(
            f"tool_call {name!r}: arguments is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise OpenAIToolCallError(
            f"tool_call {name!r}: arguments must decode to a JSON object, "
            f"got {type(parsed).__name__}"
        )
    return parsed


def _default_on_denied(error: GatewayError, tool_call: Mapping[str, Any]) -> str:
    """Default tool-message body for a policy refusal.

    Encoded as a JSON string so the model sees structured, easily-parsed
    feedback. ``tool_call`` is accepted only to match the ``on_denied``
    signature; we don't echo any of its fields here to keep the surface
    predictable.
    """
    del tool_call  # signature parity only
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
    """Render a tool result as the ``content`` string of a tool message."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return json.dumps({"value": str(result)})


def dispatch_openai_tool_call(
    wrapped: Mapping[str, Callable[..., Any]],
    tool_call: Any,
    *,
    input_label: TaintLabel | None = None,
    agent_id: str | None = None,
    resource: str | None = None,
    on_denied: Callable[[GatewayError, Mapping[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Dispatch one model-produced ``tool_call`` through the gateway.

    Returns a ``role="tool"`` message ready to be appended to the chat
    history. On allow the message contains the JSON-encoded tool result;
    on policy denial / review it contains the JSON produced by
    ``on_denied`` (default: a structured ``{"error": "policy_refusal", ...}``
    payload).

    Raises :class:`OpenAIToolCallError` for malformed input or unknown
    tool names. Re-raises any exception thrown by the underlying tool -
    those represent a tool-side bug or environmental failure that the
    surrounding orchestration loop should handle, not something the
    model should be invited to retry against.
    """
    call_id, name, arguments_str = _extract_tool_call_fields(tool_call)
    if name not in wrapped:
        raise OpenAIToolCallError(
            f"unknown tool name {name!r}; "
            f"expected one of: {sorted(wrapped)!r}"
        )
    parsed = _parse_arguments(name, arguments_str)

    forwarded: dict[str, Any] = dict(parsed)
    if input_label is not None:
        forwarded[INPUT_LABEL_KWARG] = input_label
    if agent_id is not None:
        forwarded[AGENT_ID_KWARG] = agent_id
    if resource is not None:
        forwarded[RESOURCE_KWARG] = resource
    forwarded[CALL_ID_KWARG] = call_id

    try:
        result = wrapped[name](**forwarded)
    except (PolicyDenied, PolicyReview) as exc:
        body = (on_denied or _default_on_denied)(
            exc,
            tool_call if isinstance(tool_call, Mapping) else {"id": call_id, "name": name},
        )
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": body,
        }

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": _result_to_content(result),
    }


def dispatch_openai_tool_calls(
    wrapped: Mapping[str, Callable[..., Any]],
    tool_calls: Iterable[Any],
    *,
    input_label: TaintLabel | None = None,
    agent_id: str | None = None,
    resource: str | None = None,
    on_denied: Callable[[GatewayError, Mapping[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Dispatch each ``tool_call`` in ``tool_calls`` and collect the tool messages.

    Convenience wrapper around :func:`dispatch_openai_tool_call`; the
    same ``input_label`` / ``agent_id`` / ``resource`` apply to every
    call in the batch. Callers that need to vary these per-call should
    iterate manually.
    """
    return [
        dispatch_openai_tool_call(
            wrapped,
            tc,
            input_label=input_label,
            agent_id=agent_id,
            resource=resource,
            on_denied=on_denied,
        )
        for tc in tool_calls
    ]


__all__ = [
    "OpenAITool",
    "OpenAIToolCallError",
    "dispatch_openai_tool_call",
    "dispatch_openai_tool_calls",
    "openai_tool_specs",
    "wrap_openai_tools",
]
