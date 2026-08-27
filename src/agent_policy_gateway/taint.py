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

Since R51 labels carry two dimensions (confidentiality and integrity);
every operation here is per-dimension, with the legacy single ``sources``
set counting in both, so all-legacy inputs behave exactly as before.

These functions are deliberately free of any I/O. The reference monitor
in :mod:`agent_policy_gateway.gateway` (R4) calls them; tests can call
them too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from agent_policy_gateway.core import Provenance, ProvenanceEntry, TaintLabel


def join(*labels: TaintLabel) -> TaintLabel:
    """Return the least upper bound of ``labels`` on the taint lattice.

    With zero arguments returns the bottom element (the empty label).
    With any number of arguments the operation is associative,
    commutative, and idempotent. The join is per-dimension (R51): the
    confidentiality and integrity sets union independently, with the
    legacy ``sources`` set counting in both.
    """
    out = TaintLabel()
    for lbl in labels:
        out = out.join(lbl)
    return out


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

    ``adds`` and ``declassifies`` are themselves :class:`TaintLabel`
    values, so both are dimension-aware (R51): a source in the label's
    legacy ``sources`` set adds/strips in *both* dimensions (the pre-R51
    behavior), one in its ``confidentiality`` set acts on the
    confidentiality dimension only, and one in its ``integrity`` set on
    the integrity dimension only.

    Attributes:
        adds: Sources the tool contributes on every call. ``web_search``
            adds ``{"web"}``; a CRM read adds ``{"crm.contact.email"}``.
            A dimension-scoped add marks e.g. a web reader's output as
            untrusted (integrity) without also calling it secret.
        declassifies: Sources the tool is trusted to strip from its
            output. Default: empty. Stripping a source from the
            confidentiality dimension is *declassification* proper (a
            vetted PII redactor); stripping from the integrity dimension
            is *endorsement* in IFC terms (a sanitizer vouching that
            untrusted content can no longer steer the agent).
    """

    adds: TaintLabel = field(default_factory=TaintLabel)
    declassifies: TaintLabel = field(default_factory=TaintLabel)

    @classmethod
    def of(
        cls,
        *,
        adds: Iterable[str] = (),
        declassifies: Iterable[str] = (),
        adds_confidentiality: Iterable[str] = (),
        adds_integrity: Iterable[str] = (),
        declassifies_confidentiality: Iterable[str] = (),
        declassifies_integrity: Iterable[str] = (),
    ) -> ToolTaintSpec:
        """Convenience constructor accepting plain string iterables.

        ``adds`` / ``declassifies`` act on both dimensions (the legacy
        behavior); the ``*_confidentiality`` / ``*_integrity`` kwargs
        scope the add or strip to a single dimension.
        """
        return cls(
            adds=TaintLabel.of_dimensions(
                both=adds,
                confidentiality=adds_confidentiality,
                integrity=adds_integrity,
            ),
            declassifies=TaintLabel.of_dimensions(
                both=declassifies,
                confidentiality=declassifies_confidentiality,
                integrity=declassifies_integrity,
            ),
        )


def propagate(
    input_labels: Iterable[TaintLabel],
    spec: ToolTaintSpec | None = None,
) -> TaintLabel:
    """Compute the output taint label for a tool call.

    The rule is, per dimension (R51),

        output_dim = ((∨ input_labels) ∨ spec.adds)_dim \\ spec.declassifies_dim

    where ``_dim`` is the dimension's *effective* set (legacy sources
    count in both dimensions), so a spec declassifying a source in only
    its confidentiality set leaves the source's integrity taint intact
    (and vice versa — endorsement leaves secrecy intact). The result is
    returned in canonical form, so an all-legacy input with an all-legacy
    spec produces exactly the pre-R51 single-set answer.

    With no spec (``spec=None``) the rule degenerates to a pure join
    over ``input_labels`` — i.e. the tool is treated as a transparent
    propagator with no intrinsic sources and no declassification.
    """
    spec = spec or ToolTaintSpec()
    raised = join_all(input_labels).join(spec.adds)
    if spec.declassifies.is_empty():
        return raised
    return TaintLabel(
        confidentiality=raised.confidentiality_sources
        - spec.declassifies.confidentiality_sources,
        integrity=raised.integrity_sources - spec.declassifies.integrity_sources,
    )


def propagate_provenance(
    input_provs: Iterable[Provenance],
    spec: ToolTaintSpec | None = None,
    *,
    tool_name: str,
    call_id: str | None = None,
    output_label: TaintLabel,
) -> Provenance:
    """Compute the provenance chain for a tool call's output.

    The side-channel companion to :func:`propagate`: where :func:`propagate`
    computes *what* sources flow out, this computes *where each surviving
    source came from*.

    The rule mirrors the label rule:

    1. Merge the provenance chains of all inputs (the taint carried in).
    2. Stamp a fresh :class:`ProvenanceEntry` for every source the tool
       ``adds`` — this call is the origin of those sources.
    3. Restrict the merged chain to the sources that actually survive in
       ``output_label`` — so a declassified source drops its provenance too.

    ``output_label`` is passed in (rather than recomputed) so the caller can
    reuse the label it already derived from :func:`propagate` and the two
    stay consistent.
    """
    spec = spec or ToolTaintSpec()
    merged = Provenance()
    for prov in input_provs:
        merged = merged.merge(prov)
    for source in sorted(spec.adds.all_sources):
        merged = merged.add(
            ProvenanceEntry(source=source, tool_name=tool_name, call_id=call_id)
        )
    return merged.restrict_to(output_label.all_sources)


__all__ = [
    "ToolTaintSpec",
    "flows_to",
    "join",
    "join_all",
    "propagate",
    "propagate_provenance",
    "subsumes",
]
