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
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """The gateway's decision class for a tool call."""

    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"
    REDACT = "redact"


@dataclass(frozen=True)
class TaintLabel:
    """A dual-dimension set-of-sources IFC label (R51).

    A label tracks two distinct dimensions:

    * **confidentiality** — sources marking *secret* data that must not
      reach public sinks (e.g. ``pii``, ``crm.contact.email``).
    * **integrity** — sources marking *untrusted* data that must not
      drive privileged actions (e.g. ``web``, ``agentdojo:untrusted``).

    The legacy ``sources`` set counts in **both** dimensions, so every
    pre-R51 label, policy, and audit record keeps its exact behavior.
    Labels are kept in a canonical form: a source present in both
    dimension sets is promoted to ``sources`` (the two representations
    are semantically identical), and a source in ``sources`` never also
    appears in a dimension set. Equality is therefore semantic.

    Labels form a join-semilattice per dimension: ``join`` is the
    per-dimension union. ``subsumes(other)`` is True iff ``other``'s
    effective set is a subset of this label's, in *both* dimensions.
    """

    sources: frozenset[str] = field(default_factory=frozenset)
    confidentiality: frozenset[str] = field(default_factory=frozenset)
    integrity: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Canonicalize: a source in both dimension sets means "both
        # dimensions", which is what membership in `sources` means; and a
        # source already in `sources` is redundant in a dimension set.
        promoted = self.sources | (self.confidentiality & self.integrity)
        object.__setattr__(self, "sources", promoted)
        object.__setattr__(self, "confidentiality", self.confidentiality - promoted)
        object.__setattr__(self, "integrity", self.integrity - promoted)

    @classmethod
    def of(cls, *sources: str) -> TaintLabel:
        """Construct a both-dimensions label from a varargs list of sources."""
        return cls(frozenset(sources))

    @classmethod
    def of_dimensions(
        cls,
        *,
        both: Iterable[str] = (),
        confidentiality: Iterable[str] = (),
        integrity: Iterable[str] = (),
    ) -> TaintLabel:
        """Construct a label from per-dimension source iterables."""
        return cls(
            sources=frozenset(both),
            confidentiality=frozenset(confidentiality),
            integrity=frozenset(integrity),
        )

    @property
    def all_sources(self) -> frozenset[str]:
        """Every source present in *any* dimension (the union)."""
        return self.sources | self.confidentiality | self.integrity

    @property
    def confidentiality_sources(self) -> frozenset[str]:
        """The effective confidentiality set (legacy sources count here)."""
        return self.sources | self.confidentiality

    @property
    def integrity_sources(self) -> frozenset[str]:
        """The effective integrity set (legacy sources count here)."""
        return self.sources | self.integrity

    def join(self, other: TaintLabel) -> TaintLabel:
        """Return the join (per-dimension union) of this label and ``other``."""
        return TaintLabel(
            sources=self.sources | other.sources,
            confidentiality=self.confidentiality | other.confidentiality,
            integrity=self.integrity | other.integrity,
        )

    def subsumes(self, other: TaintLabel) -> bool:
        """True iff ``other``'s effective set is contained in this label's,
        in both dimensions."""
        return (
            other.confidentiality_sources <= self.confidentiality_sources
            and other.integrity_sources <= self.integrity_sources
        )

    def without(self, remove: Iterable[str]) -> TaintLabel:
        """Return this label with ``remove`` stripped from every dimension."""
        gone = frozenset(remove)
        return TaintLabel(
            sources=self.sources - gone,
            confidentiality=self.confidentiality - gone,
            integrity=self.integrity - gone,
        )

    def is_empty(self) -> bool:
        """True for the bottom element of the lattice (no sources anywhere)."""
        return not (self.sources or self.confidentiality or self.integrity)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sources": sorted(self.sources)}
        # Serialized only when present so legacy records keep their shape.
        if self.confidentiality:
            out["confidentiality"] = sorted(self.confidentiality)
        if self.integrity:
            out["integrity"] = sorted(self.integrity)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaintLabel:
        return cls(
            sources=frozenset(d.get("sources", [])),
            confidentiality=frozenset(d.get("confidentiality", [])),
            integrity=frozenset(d.get("integrity", [])),
        )


@dataclass(frozen=True)
class ProvenanceEntry:
    """A single record of where a taint source entered an information flow.

    ``source`` is the taint label string (e.g. ``"web"``); ``tool_name`` and
    ``call_id`` identify the tool call that introduced it. ``call_id`` is the
    same id carried on the :class:`ToolCall`, so an auditor can pivot from a
    provenance entry straight back to the originating record in the log.
    """

    source: str
    tool_name: str
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "tool_name": self.tool_name,
            "call_id": self.call_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceEntry:
        return cls(
            source=str(d["source"]),
            tool_name=str(d["tool_name"]),
            call_id=d.get("call_id"),
        )


@dataclass(frozen=True)
class Provenance:
    """A side-channel taint provenance chain.

    A :class:`TaintLabel` answers *what* sources a value carries; a
    ``Provenance`` answers *where each source came from*. It is kept as a
    separate value (not folded into :class:`TaintLabel`) so the lattice
    algebra and label equality used throughout the gateway stay unchanged.

    The chain is an ordered tuple of :class:`ProvenanceEntry` records: the
    first entry for a given source is its origin, later entries record hops
    that re-introduced the same source. Duplicate entries are dropped on
    construction so merging is idempotent.
    """

    entries: tuple[ProvenanceEntry, ...] = ()

    def __post_init__(self) -> None:
        seen: set[tuple[str, str, str | None]] = set()
        deduped: list[ProvenanceEntry] = []
        for e in self.entries:
            key = (e.source, e.tool_name, e.call_id)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        object.__setattr__(self, "entries", tuple(deduped))

    def is_empty(self) -> bool:
        return not self.entries

    def add(self, entry: ProvenanceEntry) -> Provenance:
        """Return a new provenance with ``entry`` appended (deduped)."""
        return Provenance(self.entries + (entry,))

    def merge(self, other: Provenance) -> Provenance:
        """Return the union of this chain and ``other``, order-preserving."""
        return Provenance(self.entries + other.entries)

    def origins(self, source: str) -> tuple[ProvenanceEntry, ...]:
        """Return every entry recorded for ``source`` in chain order."""
        return tuple(e for e in self.entries if e.source == source)

    def restrict_to(self, sources: frozenset[str] | set[str]) -> Provenance:
        """Drop entries whose source is not in ``sources``.

        Used after declassification so a stripped source carries no
        lingering provenance into the output.
        """
        keep = set(sources)
        return Provenance(tuple(e for e in self.entries if e.source in keep))

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(
            tuple(ProvenanceEntry.from_dict(e) for e in d.get("entries", []))
        )


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
    input_provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "input_label": self.input_label.to_dict(),
            "agent_id": self.agent_id,
            "call_id": self.call_id,
        }
        # Serialized only when present so legacy records keep their shape.
        if not self.input_provenance.is_empty():
            out["input_provenance"] = self.input_provenance.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(
            tool_name=d["tool_name"],
            args=dict(d.get("args", {})),
            input_label=TaintLabel.from_dict(d.get("input_label") or {}),
            agent_id=d.get("agent_id"),
            call_id=d.get("call_id"),
            input_provenance=Provenance.from_dict(d.get("input_provenance") or {}),
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
    redacted_fields: tuple[str, ...] = ()
    output_provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": self.verdict.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "output_label": self.output_label.to_dict(),
        }
        # Serialized only when redaction actually happened, so legacy records
        # and the common allow/deny path keep their original shape.
        if self.redacted_fields:
            out["redacted_fields"] = list(self.redacted_fields)
        # Serialized only when present so legacy/common records keep their shape.
        if not self.output_provenance.is_empty():
            out["output_provenance"] = self.output_provenance.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Decision:
        return cls(
            verdict=Verdict(d["verdict"]),
            rule_id=d.get("rule_id"),
            reason=d.get("reason", ""),
            output_label=TaintLabel.from_dict(d.get("output_label") or {}),
            redacted_fields=tuple(d.get("redacted_fields", ())),
            output_provenance=Provenance.from_dict(d.get("output_provenance") or {}),
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
    "Provenance",
    "ProvenanceEntry",
    "TaintLabel",
    "ToolCall",
    "Verdict",
    "from_json",
    "to_json",
]
