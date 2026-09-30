"""Contract test: APG's TraceSig export is directly scannable by TraceSig.

APG (prevention) and TraceSig (detection) are separate projects that meet at
the ``apg-audit-trace`` format. This test runs the real TraceSig engine over
the committed example exports and pins what its bundled ``core`` and ``apg``
rule packs must find, so a change on either side that breaks the pairing
fails here. Skipped when TraceSig (0.2 or later) is not installed:

    pip install "tracesig>=0.2"
"""

from __future__ import annotations

from pathlib import Path

import pytest

tracesig = pytest.importorskip("tracesig")

if tuple(int(p) for p in tracesig.__version__.split(".")[:2]) < (0, 2):
    pytest.skip("needs tracesig >= 0.2", allow_module_level=True)

from tracesig.engine import load_rules, pack_dirs, scan  # noqa: E402

from examples.traces import SCENARIOS  # noqa: E402

TRACES_DIR = Path(__file__).resolve().parent.parent / "examples" / "traces"


def _findings(name: str) -> set[str]:
    events, stats = tracesig.load_events([str(TRACES_DIR / f"{name}.tracesig.jsonl")])
    assert stats["formats"] == ["apg"]
    assert {e.session_id for e in events} == {name}
    rules = load_rules(pack_dirs(["core", "apg"]))
    return {f.rule_id for f in scan(events, rules)}


def test_every_scenario_is_read_as_one_session() -> None:
    for name in SCENARIOS:
        _findings(name)


def test_benign_session_is_quiet() -> None:
    assert _findings("benign-session") == set()


def test_denied_injection_is_detected_even_though_it_was_blocked() -> None:
    # web content in, email out: TraceSig reports the attempt; the event's
    # verdict (deny) shows the gateway stopped it.
    assert "TS-EXF-001" in _findings("denied-injection")


def test_laundering_chain_surfaces_the_declassification() -> None:
    assert "APG-004" in _findings("laundering-chain")


def test_labels_match_what_tracesig_derives_itself() -> None:
    """APG's exported ``labels`` equal TraceSig's own derivation for old exports."""
    import json

    from tracesig.normalize import apg as tracesig_apg

    for name in SCENARIOS:
        path = TRACES_DIR / f"{name}.tracesig.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            assert event["labels"] == tracesig_apg.derive_labels(event)
