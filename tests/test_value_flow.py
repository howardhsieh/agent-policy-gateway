"""Tests for the per-value taint ledger (R57).

The :class:`ValueLedger` is the value-flow API the R57 acceptance names:
executed tool outputs are recorded with their per-value labels, later
arguments equal to a recorded value carry that value's label, and the
adapter threads the result onto ``ToolCall.arg_labels`` for
``Selector.arg_taint`` policy sub-conditions (integration covered in
``tests/test_agentdojo_adapter.py`` and the comparison-benchmark tests).
"""

from __future__ import annotations

from agent_policy_gateway import TaintLabel, ValueLedger


class TestRecordAndLookup:
    def test_starts_empty(self) -> None:
        ledger = ValueLedger()
        assert ledger.is_empty()
        assert len(ledger) == 0
        assert ledger.label_of("anything").is_empty()

    def test_recorded_value_carries_its_label(self) -> None:
        ledger = ValueLedger()
        ledger.record("doc-1", TaintLabel.of("web"))
        assert ledger.label_of("doc-1") == TaintLabel.of("web")
        assert ledger.label_of("doc-2").is_empty()
        assert len(ledger) == 1

    def test_exact_match_only(self) -> None:
        ledger = ValueLedger()
        ledger.record("doc-1", TaintLabel.of("web"))
        # A derived-but-unequal value is invisible to the PoC propagation.
        assert ledger.label_of("doc-1 plus edits").is_empty()

    def test_rerecord_joins_labels(self) -> None:
        ledger = ValueLedger()
        ledger.record("v", TaintLabel.of("web"))
        ledger.record("v", TaintLabel.of_dimensions(confidentiality=("pii",)))
        label = ledger.label_of("v")
        assert "web" in label.integrity_sources
        assert "pii" in label.confidentiality_sources
        assert "pii" not in label.integrity_sources

    def test_empty_label_is_not_recorded_and_never_launders(self) -> None:
        ledger = ValueLedger()
        ledger.record("ok", TaintLabel())
        assert ledger.is_empty()
        # Re-recording an already-tainted value with an empty label (an
        # untainted tool that happens to return an equal value) does not
        # strip the recorded taint either.
        ledger.record("v", TaintLabel.of("web"))
        ledger.record("v", TaintLabel())
        assert ledger.label_of("v") == TaintLabel.of("web")

    def test_unhashable_values_are_skipped(self) -> None:
        ledger = ValueLedger()
        ledger.record(["a", "list"], TaintLabel.of("web"))
        assert ledger.is_empty()
        assert ledger.label_of(["a", "list"]).is_empty()

    def test_non_string_hashable_values_work(self) -> None:
        ledger = ValueLedger()
        ledger.record(42, TaintLabel.of("web"))
        assert ledger.label_of(42) == TaintLabel.of("web")


class TestLabelsForArgs:
    def test_only_labeled_arguments_appear(self) -> None:
        ledger = ValueLedger()
        ledger.record("secret-note", TaintLabel.of_dimensions(confidentiality=("pii",)))
        labels = ledger.labels_for_args(
            {"payload": "secret-note", "recipient": "alice", "n": 3}
        )
        assert set(labels) == {"payload"}
        assert "pii" in labels["payload"].confidentiality_sources

    def test_all_clean_args_yield_empty_mapping(self) -> None:
        ledger = ValueLedger()
        assert ledger.labels_for_args({"a": 1, "b": "x"}) == {}

    def test_unhashable_argument_values_are_ignored(self) -> None:
        ledger = ValueLedger()
        ledger.record("v", TaintLabel.of("web"))
        assert ledger.labels_for_args({"payload": ["v"]}) == {}


class TestReset:
    def test_reset_forgets_everything(self) -> None:
        ledger = ValueLedger()
        ledger.record("v", TaintLabel.of("web"))
        ledger.reset()
        assert ledger.is_empty()
        assert ledger.label_of("v").is_empty()
