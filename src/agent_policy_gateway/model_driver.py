"""Thin LLM driver interface with record/replay (R59).

The R55 harness replays *scripted* calls; R59 puts a model in the loop:
the model reads the task prompt and every tool result (including injected
attacker text and policy refusals) and decides the next tool call itself.
This module is the seam that makes that measurable and repeatable:

* a provider-neutral **driver protocol** — one method,
  :meth:`ModelDriver.next_step`, taking the conversation so far and the
  tool schemas and returning the model's next step (tool calls, or a
  final text) plus its token usage;
* a live **Anthropic driver** (lazy import; needs ``ANTHROPIC_API_KEY``);
* a **recording** wrapper that persists every (request, step) pair to a
  JSONL fixture file, keyed by a digest of the exact request; and
* a **replay** driver that plays a fixture back deterministically,
  verifying at each step that the request it is answering is the request
  that was recorded — so CI re-runs the *identical* session with no key
  and no network, and any drift in prompts/tools/harness surfaces as a
  loud :class:`ReplayMismatch` instead of silently stale numbers.

Messages are plain dicts (``{"role": ..., "content": ...}``, tool results
additionally carrying ``tool`` and ``error``) so drivers for other
providers translate at the edge, exactly like the protocol adapters.
Nothing here imports ``agentdojo`` or any provider SDK at module scope.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AnthropicDriver",
    "ModelDriver",
    "ModelStep",
    "RecordingDriver",
    "ReplayDriver",
    "ReplayMismatch",
    "ToolInvocation",
    "ToolSchema",
    "Usage",
    "request_digest",
]


@dataclass(frozen=True)
class ToolSchema:
    """One tool as presented to the model: name, description, JSON schema."""

    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ToolInvocation:
    """A tool call the model wants made: function name + keyword arguments."""

    function: str
    args: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.function:
            raise ValueError("ToolInvocation.function must be non-empty")

    @classmethod
    def of(cls, function: str, args: Mapping[str, Any] | None = None) -> ToolInvocation:
        return cls(function=function, args=tuple(sorted((args or {}).items())))

    def arg_dict(self) -> dict[str, Any]:
        return dict(self.args)

    def to_dict(self) -> dict[str, Any]:
        return {"function": self.function, "args": self.arg_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ToolInvocation:
        return cls.of(d["function"], d.get("args") or {})


@dataclass(frozen=True)
class Usage:
    """Token cost of one driver step.

    ``estimated=True`` marks counts a driver synthesized (a simulated or
    fixture-less driver) rather than read from a provider response; the
    aggregate keeps the flag so reports can never pass an estimate off as
    a measured number.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated": self.estimated,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Usage:
        return cls(
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            estimated=bool(d.get("estimated", False)),
        )


@dataclass(frozen=True)
class ModelStep:
    """The model's next step: tool calls to make, or (with none) a final text."""

    tool_calls: tuple[ToolInvocation, ...] = ()
    text: str = ""
    usage: Usage = field(default_factory=Usage)

    @property
    def is_final(self) -> bool:
        """True when the model issued no tool calls — the turn is over."""
        return not self.tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "text": self.text,
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ModelStep:
        return cls(
            tool_calls=tuple(
                ToolInvocation.from_dict(c) for c in d.get("tool_calls", ())
            ),
            text=str(d.get("text", "")),
            usage=Usage.from_dict(d.get("usage") or {}),
        )


@runtime_checkable
class ModelDriver(Protocol):
    """Anything that can act as the model in the loop.

    ``messages`` is the neutral conversation so far — dicts with ``role``
    (``system`` / ``user`` / ``assistant`` / ``tool``) and ``content``;
    ``tool`` messages also carry ``tool`` (the function name) and
    ``error`` (``None``, or the AgentDojo-convention error string,
    policy refusals included, so the model sees exactly what an agent
    would). ``tools`` is the schema of every callable tool. The driver
    returns the model's next step.
    """

    def next_step(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolSchema],
    ) -> ModelStep:
        """Produce the model's next step for this conversation state."""
        ...


def _canonical_request(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSchema]
) -> str:
    return json.dumps(
        {
            "messages": [dict(m) for m in messages],
            "tools": [t.to_dict() for t in tools],
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def request_digest(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[ToolSchema]
) -> str:
    """Stable SHA-256 hex digest of one driver request.

    Canonical JSON (sorted keys) over the neutral messages and tool
    schemas — the identity the replay driver verifies, so a fixture can
    only answer the conversation state it was recorded against.
    """
    return hashlib.sha256(_canonical_request(messages, tools).encode("utf-8")).hexdigest()


class RecordingDriver:
    """Wrap any driver and persist every (request, step) pair as JSONL.

    Each line is ``{"digest": ..., "request": {...}, "step": {...}}`` —
    the digest is what :class:`ReplayDriver` verifies; the full request
    is kept alongside so a human can read the fixture as a transcript.
    Records append, matching the audit-log convention, so one file can
    hold a whole multi-scenario run in execution order.
    """

    def __init__(self, inner: ModelDriver, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self.recorded = 0

    def next_step(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolSchema],
    ) -> ModelStep:
        step = self._inner.next_step(messages, tools)
        record = {
            "digest": request_digest(messages, tools),
            "request": {
                "messages": [dict(m) for m in messages],
                "tools": [t.to_dict() for t in tools],
            },
            "step": step.to_dict(),
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        self.recorded += 1
        return step


class ReplayMismatch(RuntimeError):
    """A replayed request differs from the recorded one (or ran past the end)."""


class ReplayDriver:
    """Play a recorded fixture back, deterministically and key-free.

    Steps are consumed strictly in recorded order; by default each
    incoming request's digest must equal the recorded one, so the replay
    is faithful end-to-end — the harness, prompts, tool schemas, and tool
    results all have to reproduce exactly for the fixture to be usable.
    ``strict=False`` skips the digest check (useful while migrating
    harness internals without invalidating recordings).
    """

    def __init__(self, path: str | Path, *, strict: bool = True) -> None:
        self._path = Path(path)
        self._strict = strict
        self._records: list[dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: malformed fixture record on line {line_no}: {exc}"
                    ) from exc
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def remaining(self) -> int:
        """Recorded steps not yet consumed."""
        return len(self._records) - self._cursor

    def next_step(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolSchema],
    ) -> ModelStep:
        if self._cursor >= len(self._records):
            raise ReplayMismatch(
                f"{self._path}: replay exhausted after {len(self._records)} step(s) "
                "but the harness asked for another"
            )
        record = self._records[self._cursor]
        if self._strict:
            digest = request_digest(messages, tools)
            if digest != record.get("digest"):
                raise ReplayMismatch(
                    f"{self._path}: request digest mismatch at step "
                    f"{self._cursor + 1}: got {digest[:12]}…, recorded "
                    f"{str(record.get('digest'))[:12]}… (the conversation "
                    "diverged from the recording)"
                )
        self._cursor += 1
        return ModelStep.from_dict(record["step"])


_ANTHROPIC_SYSTEM_ROLES = frozenset({"system"})


class AnthropicDriver:
    """Live driver over the Anthropic Messages API (lazy import).

    Translates the neutral message/tool shapes at the edge: ``system``
    messages join the system prompt, ``tool`` messages become
    ``tool_result`` blocks (a policy refusal travels as an error result,
    exactly as an agent framework would surface it), and the response's
    ``tool_use`` blocks come back as :class:`ToolInvocation`\\ s with the
    provider-reported (non-estimated) token usage. Requires the
    ``anthropic`` package and an API key (``api_key`` argument or the
    ``ANTHROPIC_API_KEY`` environment variable).
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env without the SDK
                raise ImportError(
                    "AnthropicDriver needs the 'anthropic' package "
                    "(pip install anthropic)"
                ) from exc
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "AnthropicDriver needs an API key: pass api_key=... or set "
                    "ANTHROPIC_API_KEY"
                )
            client = anthropic.Anthropic(api_key=key)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._call_counter = 0

    # -- neutral -> Anthropic ------------------------------------------------

    def _convert(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolSchema],
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role in _ANTHROPIC_SYSTEM_ROLES:
                system_parts.append(str(content))
            elif role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(m.get("call_id", "")),
                                "content": str(
                                    m["error"] if m.get("error") else content
                                ),
                                "is_error": bool(m.get("error")),
                            }
                        ],
                    }
                )
            elif role == "assistant" and m.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for c in m["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(c.get("call_id", "")),
                            "name": c["function"],
                            "input": dict(c.get("args") or {}),
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append({"role": str(role), "content": str(content)})
        tool_defs = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": dict(t.parameters) or {"type": "object", "properties": {}},
            }
            for t in tools
        ]
        return "\n\n".join(system_parts), converted, tool_defs

    def next_step(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[ToolSchema],
    ) -> ModelStep:
        system, converted, tool_defs = self._convert(messages, tools)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system or "",
            messages=converted,
            tools=tool_defs,
        )
        calls: list[ToolInvocation] = []
        texts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                calls.append(ToolInvocation.of(block.name, dict(block.input or {})))
            elif block_type == "text":
                texts.append(block.text)
        usage = getattr(response, "usage", None)
        return ModelStep(
            tool_calls=tuple(calls),
            text="\n".join(texts),
            usage=Usage(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                estimated=False,
            ),
        )
