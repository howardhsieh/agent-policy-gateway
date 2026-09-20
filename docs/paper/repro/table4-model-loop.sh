#!/usr/bin/env bash
# Table 4 — model-in-the-loop eval (R59), replayed from committed fixtures.
# Needs: pip install -e '.[agentdojo]'   (key- and network-free)
# To re-record with a real model instead:
#   ANTHROPIC_API_KEY=... python -m agent_policy_gateway.model_benchmark \
#       --mode record --provider anthropic
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m agent_policy_gateway.model_benchmark --mode replay
