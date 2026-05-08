"""Core data model for agent-policy-gateway.

This module defines the value types every other module in the project
exchanges: ``TaintLabel`` (an IFC label), ``ToolCall`` (a request from an
agent to invoke a tool), ``Verdict`` (allow/deny/review), and ``Decision``
(the gateway's verdict on a call, with reasoning).

All types are frozen dataclasses with value-based equality and
``to_dict`` / ``from_dict`` round-tripping for JSONL audit logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """The gateway's decision class for a tool call."""

    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"


@dataclass(frozen=True)
class TaintLabel:
    """A set-of-sources IFC label.

    Labels form a join-semilattice: ``join`` is the union of source sets.
    Two labels are equal iff their source sets are equal. ``subsumes(other)``
    is True iff ``other``'s sources are a subset of this label's sources.
    """

    sources: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def of(cls, *sources: str) -> TaintLabel:
        """Construct a label from a varargs list of source strings."""
        return cls(frozenset(sources))

    def join(self, other: TaintLabel) -> TaintLabel:
        """Return the join (union) of this label and ``other``."""
        return TaintLabel(self.sources | other.sources)

    def subsumes(self, other: TaintLabel) -> bool:
        """True iff every source in ``other`` is also in ``self``."""
        return other.sources <= self.sources

    def is_empty(self) -> bool:
        """True for the bottom element of the lattice (no sources)."""
        return not self.sources

    def to_dict(self) -> dict[str, Any]:
        return {"sources": sorted(self.sources)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaintLabel:
        return cls(frozenset(d.get("sources", [])))


@dataclass(frozen=True)
class ToolCall:
    """A request from an agent to invoke a tool.

    ``input_label`` is the join of taint labels on every argument value;
    the gateway computes it before the call is dispatched.
    """

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    input_label: TaintLabel = field(default_factory=TaintLabel)
    agent_id: str | None = None
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "input_label": self.input_label.to_dict(),
            "agent_id": self.agent_id,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(
            tool_name=d["tool_name"],
            args=dict(d.get("args", {})),
            input_label=TaintLabel.from_dict(d.get("input_label") or {}),
            agent_id=d.get("agent_id"),
            call_id=d.get("call_id"),
        )


@dataclass(frozen=True)
class Decision:
    """The gateway's verdict on a tool call, plus reasoning and output label.

    ``output_label`` is the taint that will be attached to the tool's output
    if the call is allowed; it joins the input label with any source labels
    contributed by the tool itself.
    """

    verdict: Verdict
    rule_id: str | None = None
    reason: str = ""
    output_label: TaintLabel = field(default_factory=TaintLabel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "output_label": self.output_label.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Decision:
        return cls(
            verdict=Verdict(d["verdict"]),
            rule_id=d.get("rule_id"),
            reason=d.get("reason", ""),
            output_label=TaintLabel.from_dict(d.get("output_label") or {}),
        )


def to_json(obj: ToolCall | Decision | TaintLabel) -> str:
    """Serialize one of the core types to a stable JSON string."""
    if not hasattr(obj, "to_dict"):
        raise TypeError(f"{type(obj).__name__} does not implement to_dict()")
    return json.dumps(obj.to_dict(), ensure_ascii=False, sort_keys=True)


def from_json(s: str, cls: type) -> Any:
    """Deserialize a JSON string back into ``cls`` via ``cls.from_dict``."""
    if not hasattr(cls, "from_dict"):
        raise TypeError(f"{cls.__name__} does not implement from_dict()")
    return cls.from_dict(json.loads(s))


__all__ = [
    "Decision",
    "TaintLabel",
    "ToolCall",
    "Verdict",
    "from_json",
    "to_json",
]
