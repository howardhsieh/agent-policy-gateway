#!/usr/bin/env bash
# Table 1a/1b — AgentDojo single-episode replay (R50) + slack chain arm (R53).
# Needs: pip install -e '.[agentdojo]'   (~1,900 episodes, ~90 s, offline)
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -m agent_policy_gateway.agentdojo_benchmark
python -m agent_policy_gateway.agentdojo_benchmark slack --policy policies/agentdojo-chain.yaml
