#!/usr/bin/env bash
# Regenerate every table in docs/paper/index.md, in order.
set -euo pipefail
cd "$(dirname "$0")"
./environment.sh
./table1-agentdojo.sh
./table2-stateful.sh
./table3-comparison.sh
./table4-model-loop.sh
./table5-overhead.sh
