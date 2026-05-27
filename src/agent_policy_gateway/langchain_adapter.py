"""LangChain / LlamaIndex tool adapter for agent-policy-gateway (R21).

Exposes the tool objects of a LangChain (or LlamaIndex) agent through a
:class:`Gateway` in a single line of code::

    from agent_policy_gateway import Gateway, TaintLabel, wrap_langchain_tools
    gateway = Gateway(policies=[...])
    tools = wrap_langchain_tools(gateway, [search_tool, send_email_tool])
    tools["search"](query="apg", apg_input_label=TaintLabel.of("user"))

This module deliberately does **not** import ``langchain`` or
``llama_index``. Each tool object is duck-typed against the surface both
frameworks share, so the adapter ships to environments that have adopted
neither package and is tested with hand-rolled fakes (mirroring the
OpenAI / Anthropic / MCP adapters).

Tool-object shape
-----------------
* **Name.** Taken from ``tool.name`` (LangChain ``BaseTool`` /
  ``StructuredTool``) or, failing that, ``tool.metadata.name``
  (LlamaIndex ``BaseTool``). Must be a non-empty string.
* **Invocation.** The first available of, in priority order:

  1. ``tool.invoke(arguments)`` — the modern LangChain Runnable surface.
  2. ``tool.run(arguments)`` — the legacy LangChain ``BaseTool`` surface.
  3. ``tool.call(**arguments)`` — the LlamaIndex ``BaseTool`` surface.
  4. ``tool.func(**arguments)`` — LangChain ``StructuredTool``'s raw fn.
  5. ``tool.fn(**arguments)`` — LlamaIndex ``FunctionTool``'s raw fn.
  6. ``tool(**arguments)`` — a plain callable.

  The two single-argument forms (``invoke`` / ``run``) receive the
  arguments dict positionally — the structured-input convention both
  LangChain methods accept. The remaining forms receive the arguments as
  keyword arguments. The adapter returns whatever the tool returns
  verbatim (e.g. a LlamaIndex ``ToolOutput``); it does not interpret it.

Everything else — the four reserved kwargs, ``taint_specs``,
``resource_args``, and ``prefix`` — behaves exactly as in
:func:`agent_policy_gateway.mcp_adapter.wrap_mcp_session`, so audit,
taint propagation, and policy enforcement are identical across adapters.
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


def _tool_name(tool: Any) -> str:
    """Extract the advertised tool name from a framework tool object.

    Accepts the LangChain shape (``tool.name``) and the LlamaIndex shape
    (``tool.metadata.name``). Raises :class:`ValueError` for anything
    without a non-empty string name so misshapen tools surface
    immediately.
    """
    name = getattr(tool, "name", None)
    if name is None:
        metadata = getattr(tool, "metadata", None)
        if metadata is not None:
            name = getattr(metadata, "name", None)
    if name is None:
        raise ValueError(
            f"tool object has no 'name' (or 'metadata.name') attribute: {tool!r}"
        )
    if not isinstance(name, str) or not name:
        raise ValueError(f"tool name must be a non-empty string, got {name!r}")
    return name


def _invoke_tool(tool: Any, *, arguments: dict[str, Any]) -> Any:
    """Invoke a duck-typed LangChain / LlamaIndex tool object.

    Tries the framework invocation surfaces in priority order (see the
    module docstring). Raises :class:`TypeError` if none is available.
    """
    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        return invoke(arguments)
    run = getattr(tool, "run", None)
    if callable(run):
        return run(arguments)
    call = getattr(tool, "call", None)
    if callable(call):
        return call(**arguments)
    func = getattr(tool, "func", None)
    if callable(func):
        return func(**arguments)
    fn = getattr(tool, "fn", None)
    if callable(fn):
        return fn(**arguments)
    if callable(tool):
        return tool(**arguments)
    raise TypeError(
        "tool object is not invocable: expected one of .invoke / .run / "
        f".call / .func / .fn / __call__, got {tool!r}"
    )


def wrap_langchain_tools(
    gateway: Gateway,
    tools: Iterable[Any],
    *,
    taint_specs: Mapping[str, ToolTaintSpec] | None = None,
    resource_args: Mapping[str, str] | None = None,
    prefix: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """Mount every LangChain / LlamaIndex ``tool`` object under ``gateway``.

    Returns a dict mapping the *registered* tool name (the advertised
    name, optionally prefixed with ``"<prefix>."``) to a gateway-mediated
    callable. Each callable accepts arbitrary keyword arguments — they
    are forwarded to the underlying tool object as its structured input —
    plus the four reserved kwargs the gateway recognises:

    * ``apg_input_label`` — the call's input :class:`TaintLabel`.
    * ``apg_agent_id`` — identity string for selector matching.
    * ``apg_call_id`` — caller-supplied id (a fresh uuid4 hex by default).
    * ``apg_resource`` — explicit resource override; takes precedence
      over any value derived from ``resource_args``.

    ``taint_specs`` keys are the *advertised* tool names; matching specs
    are registered with the gateway against the *registered* (prefixed)
    name. Tools without a spec are treated as transparent propagators.

    ``resource_args`` declares per-tool which advertised-name argument
    carries the policy resource, matched against :class:`Selector`
    ``resource``.

    ``prefix`` is joined to each advertised name with a ``"."`` separator
    so multiple tool sets can be mounted on the same gateway without name
    clashes. The underlying tool object always receives its own
    (un-prefixed) input.
    """
    specs = dict(taint_specs or {})
    res_args = dict(resource_args or {})

    wrapped: dict[str, Callable[..., Any]] = {}
    seen: set[str] = set()

    for tool in tools:
        advertised = _tool_name(tool)
        registered = f"{prefix}.{advertised}" if prefix else advertised
        if registered in seen:
            raise ValueError(
                f"duplicate tool name in binding: {registered!r}"
            )
        seen.add(registered)

        spec = specs.get(advertised)
        if spec is not None:
            gateway.register_tool(registered, spec)
        resource_arg_name = res_args.get(advertised)

        wrapped[registered] = _build_wrapper(
            gateway=gateway,
            tool=tool,
            registered_name=registered,
            resource_arg_name=resource_arg_name,
        )

    return wrapped


def _build_wrapper(
    *,
    gateway: Gateway,
    tool: Any,
    registered_name: str,
    resource_arg_name: str | None,
) -> Callable[..., Any]:
    """Return a per-tool callable that mediates one tool call through ``gateway``."""

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

        # Snapshot the arguments forwarded to the tool so that a mutating
        # audit subscriber cannot silently change what the tool sees, and
        # so the audit log records exactly what was sent.
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
            _invoke_tool,
            tool,
            arguments=arguments,
            resource=resource,
        )
        return value

    safe_name = registered_name.replace(".", "_") or "langchain_tool"
    wrapper.__name__ = safe_name
    wrapper.__qualname__ = safe_name
    wrapper.__doc__ = (
        f"Gateway-mediated call to LangChain/LlamaIndex tool {registered_name!r}."
    )
    return wrapper


__all__ = ["wrap_langchain_tools"]
