"""Taint propagation algebra for agent-policy-gateway.

This module turns the lattice operations on :class:`TaintLabel` into the
propagation rules the gateway uses at runtime. The contract is small:

* :func:`join` is the n-ary least upper bound on the source-set lattice.
  It is associative, commutative, and idempotent, with the empty label as
  the identity.
* :func:`subsumes` is the lattice order ``⊑``. ``subsumes(a, b)`` is True
  iff every source in ``b`` is also in ``a`` — i.e. ``b ⊑ a``.
* :class:`ToolTaintSpec` declares a tool's *intrinsic* sources (added on
  every call — e.g. ``web_search`` adds ``web``) and the sources it
  *declassifies* (strips off the output — e.g. a vetted PII redactor
  removing ``pii``).
* :func:`propagate` is the pure rule

      output = ((∨ inputs) ∨ spec.adds) \\ spec.declassifies

  used by the gateway to compute the label attached to a tool's output.
* :func:`flows_to` is a convenience for policy authors: True iff a label
  is permitted to flow into a sink whose allowed sources are ``allowed``.

These functions are deliberately free of any I/O. The reference monitor
in :mod:`agent_policy_gateway.gateway` (R4) calls them; tests can call
them too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from agent_policy_gateway.core import TaintLabel


def join(*labels: TaintLabel) -> TaintLabel:
    """Return the least upper bound of ``labels`` on the taint lattice.

    With zero arguments returns the bottom element (the empty label).
    With any number of arguments the operation is associative,
    commutative, and idempotent.
    """
    sources: frozenset[str] = frozenset()
    for lbl in labels:
        sources = sources | lbl.sources
    return TaintLabel(sources)


def join_all(labels: Iterable[TaintLabel]) -> TaintLabel:
    """Iterable form of :func:`join` for callers that already have a list."""
    return join(*labels)


def subsumes(higher: TaintLabel, lower: TaintLabel) -> bool:
    """Return True iff ``lower ⊑ higher`` on the lattice.

    Equivalent to ``higher.subsumes(lower)``; provided as a free function
    so policy code can read top-down: ``subsumes(allowed, observed)``.
    """
    return higher.subsumes(lower)


def flows_to(label: TaintLabel, allowed: TaintLabel) -> bool:
    """Return True iff ``label`` is permitted to flow into a sink whose
    accepted sources are ``allowed``.

    Operationally identical to ``subsumes(allowed, label)`` but named for
    the direction of information flow. ``flows_to(web, {})`` is False —
    web-tainted data cannot flow into a sink that admits no sources.
    """
    return allowed.subsumes(label)


@dataclass(frozen=True)
class ToolTaintSpec:
    """Declarative taint behaviour for a single tool.

    Attributes:
        adds: Sources the tool contributes on every call. ``web_search``
            adds ``{"web"}``; a CRM read adds ``{"crm.contact.email"}``.
        declassifies: Sources the tool is trusted to strip from its
            output. Default: empty (no declassification). A vetted
            redactor might declassify ``{"pii"}``.
    """

    adds: TaintLabel = field(default_factory=TaintLabel)
    declassifies: TaintLabel = field(default_factory=TaintLabel)

    @classmethod
    def of(
        cls,
        *,
        adds: Iterable[str] = (),
        declassifies: Iterable[str] = (),
    ) -> ToolTaintSpec:
        """Convenience constructor accepting plain string iterables."""
        return cls(
            adds=TaintLabel(frozenset(adds)),
            declassifies=TaintLabel(frozenset(declassifies)),
        )


def propagate(
    input_labels: Iterable[TaintLabel],
    spec: ToolTaintSpec | None = None,
) -> TaintLabel:
    """Compute the output taint label for a tool call.

    The rule is

        output = ((∨ input_labels) ∨ spec.adds) \\ spec.declassifies

    With no spec (``spec=None``) the rule degenerates to a pure join
    over ``input_labels`` — i.e. the tool is treated as a transparent
    propagator with no intrinsic sources and no declassification.
    """
    spec = spec or ToolTaintSpec()
    raised = join_all(input_labels).join(spec.adds)
    if spec.declassifies.is_empty():
        return raised
    return TaintLabel(raised.sources - spec.declassifies.sources)


__all__ = [
    "ToolTaintSpec",
    "flows_to",
    "join",
    "join_all",
    "propagate",
    "subsumes",
]
