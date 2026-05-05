"""MCP adapter for agent-policy-gateway (R6).

Exposes the tools of an MCP-compatible session through a :class:`Gateway`
in a single line of code::

    from agent_policy_gateway import Gateway, TaintLabel, wrap_mcp_session
    gateway = Gateway(policies=[...])
    tools = wrap_mcp_session(gateway, mcp_session)
    tools["search"](query="apg", apg_input_label=TaintLabel.of("user"))

This module deliberately does **not** import the ``mcp`` package. The
session is duck-typed against a tiny synchronous protocol::

    class MCPSessionLike(Protocol):
        def list_tools(self) -> Iterable[Any] | Any: ...
        def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

``list_tools()`` may return either an iterable of descriptors or an
object with a ``.tools`` attribute (the shape used by ``mcp.ClientSession``).
Each descriptor must expose a non-empty string ``name``, either as an
attribute or as a dict key. Anything else raises a clear
:class:`ValueError` so misshapen sessions surface immediately.

This module mediates *sync* MCP sessions. For the async ``mcp`` SDK
transport (where ``ClientSession.list_tools`` and ``call_tool`` are
``async def``), use
:func:`agent_policy_gateway.mcp_async_adapter.wrap_mcp_session_async`,
which has the same surface and semantics but awaits each MCP call
through :meth:`Gateway.aexecute`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agent_policy_gateway.core import TaintLabel, ToolCall
from agent_policy_gateway.gateway import (
    AGENT_ID_KWARG,
    CALL_ID_KWARG,
    INPUT_LABEL_KWARG,
    RESOURCE_KWARG,
    Gateway,
)
from agent_policy_gateway.taint import ToolTaintSpec


def _tool_name(descriptor: Any) -> str:
    """Extract the advertised tool name from a descriptor.

    Accepts dict-like (``descriptor["name"]``) and attribute-style
    (``descriptor.name``) shapes, matching both ad-hoc fakes and the
    ``mcp.types.Tool`` payload from the real SDK.
    """
    if isinstance(descriptor, Mapping):
        if "name" not in descriptor:
            raise ValueError(f"MCP tool descriptor missing 'name': {descriptor!r}")
        name = descriptor["name"]
    else:
        name = getattr(descriptor, "name", None)
        if name is None:
            raise ValueError(
                f"MCP tool descriptor has no 'name' attribute: {descriptor!r}"
            )
    if not isinstance(name, str) or not name:
        raise ValueError(f"MCP tool name must be a non-empty string, got {name!r}")
    return name


def _iter_tool_descriptors(list_tools_result: Any) -> Iterable[Any]:
    """Normalise either a bare iterable or a ``.tools``-bearing wrapper."""
    if hasattr(list_tools_result, "tools"):
        return list_tools_result.tools
    return list_tools_result


def wrap_mcp_session(
    gateway: Gateway,
    session: Any,
    *,
    taint_specs: Mapping[str, ToolTaintSpec] | None = None,
    resource_args: Mapping[str, str] | None = None,
    prefix: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """Mount every tool advertised by ``session`` under ``gateway``.

    Returns a dict mapping the *registered* tool name (the advertised
    name, optionally prefixed with ``"<prefix>."``) to a gateway-mediated
    callable. Each callable accepts arbitrary keyword arguments — they
    are forwarded as ``session.call_tool(name, arguments=kwargs)`` —
    plus the four reserved kwargs the gateway recognises:

    * ``apg_input_label`` — the call's input :class:`TaintLabel`.
    * ``apg_agent_id`` — identity string for selector matching.
    * ``apg_call_id`` — caller-supplied id (a fresh uuid4 hex by default).
    * ``apg_resource`` — explicit resource override; takes precedence
      over any value derived from ``resource_args``.

    ``taint_specs`` keys are the *advertised* tool names; matching specs
    are registered with the gateway against the *registered* (prefixed)
    name. Tools without a spec are treated as transparent propagators —
    the gateway propagates input taint without adding sources.

    ``resource_args`` declares per-tool which advertised-name argument
    carries the policy resource. The bound value is passed through to
    :class:`Selector` ``resource`` matching via :class:`Gateway`.

    ``prefix`` is joined to each advertised name with a ``"."`` separator
    so multiple sessions (e.g. ``filesystem`` and ``net``) can be mounted
    on the same gateway without name clashes. The underlying MCP call
    always uses the advertised (un-prefixed) name.
    """
    if not callable(getattr(session, "list_tools", None)):
        raise TypeError("session does not provide a callable list_tools()")
    if not callable(getattr(session, "call_tool", None)):
        raise TypeError("session does not provide a callable call_tool()")

    specs = dict(taint_specs or {})
    res_args = dict(resource_args or {})

    wrapped: dict[str, Callable[..., Any]] = {}
    seen: set[str] = set()

    for descriptor in _iter_tool_descriptors(session.list_tools()):
        advertised = _tool_name(descriptor)
        registered = f"{prefix}.{advertised}" if prefix else advertised
        if registered in seen:
            raise ValueError(
                f"MCP session advertises duplicate tool name: {registered!r}"
            )
        seen.add(registered)

        spec = specs.get(advertised)
        if spec is not None:
            gateway.register_tool(registered, spec)
        resource_arg_name = res_args.get(advertised)

        wrapped[registered] = _build_wrapper(
            gateway=gateway,
            session=session,
            advertised_name=advertised,
            registered_name=registered,
            resource_arg_name=resource_arg_name,
        )

    return wrapped


def _build_wrapper(
    *,
    gateway: Gateway,
    session: Any,
    advertised_name: str,
    registered_name: str,
    resource_arg_name: str | None,
) -> Callable[..., Any]:
    """Return a per-tool callable that mediates one MCP call through ``gateway``."""

    def wrapper(**kwargs: Any) -> Any:
        input_label = kwargs.pop(INPUT_LABEL_KWARG, None) or TaintLabel()
        agent_id = kwargs.pop(AGENT_ID_KWARG, None)
        call_id = kwargs.pop(CALL_ID_KWARG, None) or uuid.uuid4().hex
        explicit_resource = kwargs.pop(RESOURCE_KWARG, None)

        resource = explicit_resource
        if resource is None and resource_arg_name is not None:
            value = kwargs.get(resource_arg_name)
            if value is not None:
                resource = value if isinstance(value, str) else str(value)

        # Snapshot the arguments forwarded to the MCP server so that a
        # mutating audit subscriber cannot silently change what the tool
        # sees, and so the audit log records exactly what was sent.
        arguments = dict(kwargs)

        call = ToolCall(
            tool_name=registered_name,
            args=arguments,
            input_label=input_label,
            agent_id=agent_id,
            call_id=call_id,
        )
        value, _ = gateway.execute(
            call,
            session.call_tool,
            advertised_name,
            arguments=arguments,
            resource=resource,
        )
        return value

    safe_name = registered_name.replace(".", "_") or "mcp_tool"
    wrapper.__name__ = safe_name
    wrapper.__qualname__ = safe_name
    wrapper.__doc__ = f"Gateway-mediated call to MCP tool {advertised_name!r}."
    return wrapper


__all__ = ["wrap_mcp_session"]
