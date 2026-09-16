"""Per-value taint tracking (R57).

The session-level label the AgentDojo adapter accumulates (R49a) answers
*"has this session touched source X?"* — one bit per source for the whole
conversation. R56 measured the two residuals that granularity leaves
open: a covert attack whose sink call is observationally identical to
the legitimate flow (finding 3), and the one-for-one trade between
confidentiality coverage and secret-touching utility (finding 4). Both
residuals share a root: the policy can see *that* the session carries a
source, but not *which value* carries it.

This module adds the missing granularity as a **value ledger**: a
mapping from concrete tool-output values to the :class:`TaintLabel`
each value carries. The propagation rule is deliberately the simplest
one that demonstrates the mechanism — **exact-match**:

* when a mediated tool call executes, its output value is recorded with
  ``propagate(labels of its argument values, spec)`` — the same R51
  per-dimension rule the session label uses, applied at value scope;
* when a later call passes an argument *equal* to a recorded value,
  that argument carries the recorded label (see
  :meth:`ValueLedger.labels_for_args`), and the gateway exposes it to
  policies as ``ToolCall.arg_labels`` for ``Selector.arg_taint``
  sub-conditions.

Known PoC limits, stated up front: equality is the only derivation the
ledger can see. A value transformed in any way the runtime does not
mediate (concatenation, paraphrase, re-encoding by the model itself)
breaks the chain — the honest boundary ``docs/benchmarks/comparison.md``
spells out. Unhashable values are skipped (never recorded, never
matched), and values whose computed label is empty are not recorded, so
ubiquitous clean outputs (``"ok"``) can never alias a tainted one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_policy_gateway.core import TaintLabel

__all__ = ["ValueLedger"]


def _hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


class ValueLedger:
    """Exact-match value → :class:`TaintLabel` bookkeeping (R57).

    One ledger tracks one session; :meth:`reset` clears it between
    episodes exactly like the session label. The ledger is deliberately
    conservative in both directions:

    * **join on re-record** — a value produced twice with different
      labels keeps the join, never the latest (a value that *ever*
      carried a source still carries it);
    * **empty labels are not stored** — an untainted output neither
      grows the ledger nor (by aliasing an equal tainted value later)
      launders it: recording ``v`` with an empty label leaves any
      existing entry for ``v`` untouched.
    """

    def __init__(self) -> None:
        self._labels: dict[Any, TaintLabel] = {}

    def __len__(self) -> int:
        return len(self._labels)

    def is_empty(self) -> bool:
        """True iff nothing labeled has been recorded (or after :meth:`reset`)."""
        return not self._labels

    def record(self, value: Any, label: TaintLabel) -> None:
        """Remember that ``value`` carries ``label``.

        Unhashable values and empty labels are skipped (see the class
        docstring); a re-recorded value keeps the join of every label it
        was ever recorded with.
        """
        if label.is_empty() or not _hashable(value):
            return
        existing = self._labels.get(value)
        self._labels[value] = label if existing is None else existing.join(label)

    def label_of(self, value: Any) -> TaintLabel:
        """The recorded label for ``value`` (empty when unknown/unhashable)."""
        if not _hashable(value):
            return TaintLabel()
        return self._labels.get(value, TaintLabel())

    def labels_for_args(self, args: Mapping[str, Any]) -> dict[str, TaintLabel]:
        """Per-argument labels for a call's ``args``, empty labels omitted.

        The result is what the adapter stamps on ``ToolCall.arg_labels``:
        only arguments whose value the ledger knows appear, so an
        all-clean call keeps the pre-R57 record shape.
        """
        out: dict[str, TaintLabel] = {}
        for name, value in args.items():
            label = self.label_of(value)
            if not label.is_empty():
                out[name] = label
        return out

    def reset(self) -> None:
        """Forget every recorded value (call between sessions/episodes)."""
        self._labels.clear()
