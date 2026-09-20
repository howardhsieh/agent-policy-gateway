#!/usr/bin/env bash
# Table 3a/3b — seven-arm paradigm comparison (R56/R57). No extras needed.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m agent_policy_gateway.comparison_benchmark
