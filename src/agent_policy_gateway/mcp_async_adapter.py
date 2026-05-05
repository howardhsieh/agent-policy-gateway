"""Async MCP adapter for agent-policy-gateway (R9).

Sibling of :mod:`agent_policy_gateway.mcp_adapter` for the async MCP
transport. The real ``mcp`` SDK ships ``async def list_tools()`` and
``async def call_tool(name, arguments)`` on its ``ClientSession``;
this module duck-types that protocol so the SDK is *not* a runtime
dependency of ``agent-policy-gateway``::

    class AsyncMCPSessionLike(Protocol):
        async def list_tools(self) -> Iterable[Any] | Any: ...
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

Usage::

    from agent_policy_gateway import Gateway, TaintLabel, wrap_mcp_session_async

    gateway = Gateway(policies=[...])
    tools = await wrap_mcp_session_async(gateway, mcp_session)
    result = await tools["search"](query="apg", apg_input_label=TaintLabel.of("user"))

The discovery call (``await session.list_tools()``) is performed
eagerly during :func:`wrap_mcp_session_async` so per-tool callables
can be built up front; each per-tool callable is itself
``async def`` and awaits ``session.call_tool`` through the gateway's
:meth:`Gateway.aexecute` path.

Selector resource binding, ``prefix`` namespacing, ``taint_specs``
registration, and the four reserved kwargs (``apg_input_label``,
``apg_agent_id``, ``apg_call_id``, ``apg_resource``) all match the
synchronous adapter exactly. Misshapen sessions raise
:class:`TypeError` / :class:`ValueError` immediately.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agent_policy_gateway.core import TaintLabel, ToolCall
from agent_policy_gateway.gateway import (
    AGENT_ID_KWARG,
    CALL_ID_KWARG,
    INPUT_LABEL_KWARG,
    RESOURCE_KWARG,
    Gateway,
)
from agent_policy_gateway.mcp_adapter import _iter_tool_descriptors, _tool_name
from agent_policy_gateway.taint import ToolTaintSpec


async def wrap_mcp_session_async(
    gateway: Gateway,
    session: Any,
    *,
    taint_specs: Mapping[str, ToolTaintSpec] | None = None,
    resource_args: Mapping[str, str] | None = None,
    prefix: str | None = None,
) -> dict[str, Callable[..., Awaitable[Any]]]:
    """Mount every tool advertised by an *async* MCP-style session.

    Returns a dict mapping the *registered* tool name (the advertised
    name, optionally prefixed with ``"<prefix>."``) to an ``async def``
    callable that mediates one MCP call through ``gateway.aexecute``.

    The function itself is ``async`` because ``session.list_tools()`` is
    awaited as part of discovery. Each returned per-tool callable
    accepts arbitrary keyword arguments and forwards them as
    ``await session.call_tool(advertised_name, arguments=kwargs)``,
    plus the same reserved kwargs the gateway recognises elsewhere
    (``apg_input_label``, ``apg_agent_id``, ``apg_call_id``,
    ``apg_resource``).

    Semantics — including ``taint_specs`` keying on advertised names,
    duplicate-name detection, and ``apg_resource`` overriding any
    ``resource_args`` derivation — match
    :func:`agent_policy_gateway.mcp_adapter.wrap_mcp_session` exactly.
    The only difference is that ``list_tools`` and ``call_tool`` are
    awaited.
    """
    if not callable(getattr(session, "list_tools", None)):
        raise TypeError("session does not provide a callable list_tools()")
    if not callable(getattr(session, "call_tool", None)):
        raise TypeError("session does not provide a callable call_tool()")

    specs = dict(taint_specs or {})
    res_args = dict(resource_args or {})

    list_result = session.list_tools()
    if inspect.isawaitable(list_result):
        list_result = await list_result

    wrapped: dict[str, Callable[..., Awaitable[Any]]] = {}
    seen: set[str] = set()

    for descriptor in _iter_tool_descriptors(list_result):
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

        wrapped[registered] = _build_async_wrapper(
            gateway=gateway,
            session=session,
            advertised_name=advertised,
            registered_name=registered,
            resource_arg_name=resource_arg_name,
        )

    return wrapped


def _build_async_wrapper(
    *,
    gateway: Gateway,
    session: Any,
    advertised_name: str,
    registered_name: str,
    resource_arg_name: str | None,
) -> Callable[..., Awaitable[Any]]:
    """Return a per-tool ``async`` callable mediating one MCP call."""

    async def wrapper(**kwargs: Any) -> Any:
        input_label = kwargs.pop(INPUT_LABEL_KWARG, None) or TaintLabel()
        agent_id = kwargs.pop(AGENT_ID_KWARG, None)
        call_id = kwargs.pop(CALL_ID_KWARG, None) or uuid.uuid4().hex
        explicit_resource = kwargs.pop(RESOURCE_KWARG, None)

        resource = explicit_resource
        if resource is None and resource_arg_name is not None:
            value = kwargs.get(resource_arg_name)
            if value is not None:
                resource = value if isinstance(value, str) else str(value)

        # Snapshot the arguments forwarded to the MCP server so a
        # mutating audit subscriber cannot silently change what the tool
        # sees, and so the audit log records exactly what was sent.
        arguments: dict[str, Any] = dict(kwargs)

        call = ToolCall(
            tool_name=registered_name,
            args=arguments,
            input_label=input_label,
            agent_id=agent_id,
            call_id=call_id,
        )
        value, _ = await gateway.aexecute(
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
    wrapper.__doc__ = (
        f"Async gateway-mediated call to MCP tool {advertised_name!r}."
    )
    return wrapper


__all__ = ["wrap_mcp_session_async"]
