#!/usr/bin/env bash
# Table 2 — long-horizon stateful laundering family (R55). No extras needed.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m agent_policy_gateway.stateful_benchmark
