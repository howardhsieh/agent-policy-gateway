#!/usr/bin/env bash
# Table 5 — gateway overhead micro-benchmark (R60), seed 0.
# Absolute latencies are machine-specific; structure and growth curves are not.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m agent_policy_gateway.overhead_benchmark --seed 0
