"""R61: keep the paper draft (``docs/paper/index.md``) honest.

The draft's tables are prose copies of benchmark output, and prose
copies drift. These tests re-derive every deterministic table
in-process and compare it cell-by-cell against the markdown, pin the
machine-specific / extras-gated tables against their source benchmark
pages, and check that each table's committed reproducibility script
exists, parses, and references real modules.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "docs" / "paper" / "index.md"
REPRO = REPO_ROOT / "docs" / "paper" / "repro"
OVERHEAD_DOC = REPO_ROOT / "docs" / "benchmarks" / "overhead.md"

REPRO_SCRIPTS = (
    "table1-agentdojo.sh",
    "table2-stateful.sh",
    "table3-comparison.sh",
    "table4-model-loop.sh",
    "table5-overhead.sh",
    "all.sh",
    "environment.sh",
)


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _paper_text() -> str:
    return PAPER.read_text(encoding="utf-8")


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown pipe-table in ``text`` as (header, data rows)."""
    tables: list[tuple[list[str], list[list[str]]]] = []
    block: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue  # separator row
            block.append(cells)
        elif block:
            tables.append((block[0], block[1:]))
            block = []
    if block:
        tables.append((block[0], block[1:]))
    return tables


def _find_table(
    text: str, *, required_headers: tuple[str, ...], rows: int | None = None
) -> tuple[list[str], list[list[str]]]:
    """The unique paper table whose header contains ``required_headers``."""
    matches = [
        (header, data)
        for header, data in _tables(text)
        if all(h in header for h in required_headers)
        and (rows is None or len(data) == rows)
    ]
    assert len(matches) == 1, (
        f"expected exactly one table with headers {required_headers} "
        f"({rows} rows), found {len(matches)}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# Reproducibility scripts


def test_every_repro_script_exists_and_is_executable() -> None:
    for name in REPRO_SCRIPTS:
        script = REPRO / name
        assert script.is_file(), f"missing {script}"
        assert os.access(script, os.X_OK), f"{name} is not executable"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_every_repro_script_parses() -> None:
    for name in REPRO_SCRIPTS:
        proc = subprocess.run(
            ["bash", "-n", str(REPRO / name)], capture_output=True, text=True
        )
        assert proc.returncode == 0, f"bash -n {name}: {proc.stderr}"


def test_repro_scripts_reference_real_modules() -> None:
    for name in REPRO_SCRIPTS:
        body = (REPRO / name).read_text(encoding="utf-8")
        for module in re.findall(r"python -m ([\w.]+)", body):
            # find_spec locates without importing, so agentdojo-gated
            # modules resolve here even without the extra installed.
            assert importlib.util.find_spec(module) is not None, (
                f"{name} references unknown module {module}"
            )


def test_paper_names_every_table_script() -> None:
    text = _paper_text()
    for name in REPRO_SCRIPTS:
        assert name in text, f"paper never mentions repro script {name}"


def test_requirements_lock_committed() -> None:
    lock = REPRO / "requirements.lock"
    body = lock.read_text(encoding="utf-8")
    assert "agentdojo" in body
    assert len(body.splitlines()) > 20


# ---------------------------------------------------------------------------
# Table 1 (AgentDojo) — pinned against the benchmark page's figures. The
# banking figures are additionally re-derived from the live matrix by
# ``tests/test_agentdojo_benchmark.py``; re-running all ~1,900 episodes
# here would double that suite's cost for no extra signal.

TABLE1A_PINNED = {
    ("banking", "no-defense"): ["144", "100.0%", "144", "100.0%", "100.0%", "0"],
    ("banking", "apg"): ["144", "25.0%", "144", "0.0%", "11.1%", "284"],
    ("slack", "no-defense"): ["105", "100.0%", "84", "100.0%", "100.0%", "0"],
    ("slack", "apg"): ["105", "4.8%", "84", "0.0%", "60.0%", "296"],
    ("travel", "no-defense"): ["140", "100.0%", "120", "100.0%", "100.0%", "0"],
    ("travel", "apg"): ["140", "70.0%", "120", "0.0%", "50.0%", "162"],
    ("workspace", "no-defense"): ["560", "100.0%", "240", "100.0%", "100.0%", "0"],
    ("workspace", "apg"): ["560", "55.0%", "240", "0.0%", "50.0%", "560"],
}
TABLE1B_PINNED = {
    ("slack", "apg-chain"): ["105", "4.8%", "84", "0.0%", "40.0%", "382"],
}


def test_table1_matches_pinned_agentdojo_numbers() -> None:
    text = _paper_text()
    table1a = _find_table(text, required_headers=("suite", "ASR (sink)"), rows=8)
    table1b = _find_table(text, required_headers=("suite", "ASR (sink)"), rows=1)
    for (header, data), pinned in ((table1a, TABLE1A_PINNED), (table1b, TABLE1B_PINNED)):
        assert header[:2] == ["suite", "arm"]
        got = {(row[0], row[1]): row[2:] for row in data}
        assert got == pinned


# ---------------------------------------------------------------------------
# Table 2 (stateful) — re-derived in-process.


def test_table2_matches_stateful_benchmark() -> None:
    from agent_policy_gateway.stateful_benchmark import run_demo

    _, data = _find_table(_paper_text(), required_headers=("arm", "persist"))
    got = {row[0]: row[1:] for row in data}
    expected = {}
    for summary in run_demo(policy_dir=REPO_ROOT):
        expected[summary["arm"]] = [
            str(summary["scenarios"]),
            _pct(summary["utility"]),
            _pct(summary["compromise_rate"]),
            _pct(summary["compromise_direct"]),
            _pct(summary["compromise_launder"]),
            str(summary["refused_calls"]),
            f"{summary['mean_taint_persistence']:.2f}",
        ]
    assert got == expected


# ---------------------------------------------------------------------------
# Table 3 (comparison) — re-derived in-process, both matrices.


def test_table3_matches_comparison_benchmark() -> None:
    from agent_policy_gateway.comparison_benchmark import (
        ATTACK_VARIANTS,
        BENIGN_VARIANTS,
        run_comparison,
    )

    text = _paper_text()
    _, attack_rows = _find_table(text, required_headers=("arm", "overt-direct"))
    _, benign_rows = _find_table(text, required_headers=("arm", "clean"))
    got_attack = {row[0]: row[1:] for row in attack_rows}
    got_benign = {row[0]: row[1:] for row in benign_rows}

    expected_attack = {}
    expected_benign = {}
    for summary in run_comparison(policy_dir=REPO_ROOT):
        arm = summary["arm"]
        expected_attack[arm] = [
            _pct(summary["utility"]),
            _pct(summary["compromise_rate"]),
            *(_pct(summary["compromise_by_variant"][v]) for v in ATTACK_VARIANTS),
        ]
        expected_benign[arm] = [
            _pct(summary["utility"]),
            *(_pct(summary["utility_by_variant"][v]) for v in BENIGN_VARIANTS),
        ]
    assert got_attack == expected_attack
    assert got_benign == expected_benign


# ---------------------------------------------------------------------------
# Table 4 (model loop) — re-derived from the committed replay fixtures.


def test_table4_matches_model_replay() -> None:
    pytest.importorskip("agentdojo")
    from agent_policy_gateway.model_benchmark import run_matrix

    _, data = _find_table(_paper_text(), required_headers=("arm", "reaction"))
    got = {(row[0], row[1]): row[2:] for row in data}
    expected = {}
    for row in run_matrix(
        mode="replay",
        fixtures_dir=REPO_ROOT / "examples" / "model_loop" / "fixtures",
        policy_dir=REPO_ROOT,
    ):
        stats = row["stats"]
        reaction = "—" if row["arm"] == "no-defense" else row["reaction"]
        tokens = f"{stats.total_tokens:,}"
        expected[(row["arm"], reaction)] = [
            _pct(stats.utility),
            _pct(stats.attack_success_rate),
            str(stats.refused_calls),
            str(stats.retried_refusals),
            str(stats.driver_steps),
            f"~{tokens}" if stats.tokens_estimated else tokens,
        ]
    assert got == expected


# ---------------------------------------------------------------------------
# Table 5 (overhead) — machine-specific latencies; pinned against the
# benchmark page so the paper and docs/benchmarks/overhead.md cannot drift
# apart. (The structure itself is asserted machine-independently by
# ``tests/test_overhead_benchmark.py``.)


def _ladder_cells(path: Path) -> dict[str, list[str]]:
    """Rung name -> numeric cells, from a file's mediation-ladder table."""
    text = path.read_text(encoding="utf-8")
    header, data = _find_table(text, required_headers=("rung",))
    assert header[-1] == "overhead µs"
    out = {}
    for row in data:
        match = re.search(r"`([a-z_]+)`", row[0])
        assert match, f"unlabelled ladder row: {row[0]}"
        out[match.group(1)] = row[1:]
    return out


def test_table5_matches_overhead_page() -> None:
    assert _ladder_cells(PAPER) == _ladder_cells(OVERHEAD_DOC)
