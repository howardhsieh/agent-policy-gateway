#!/usr/bin/env bash
# Report the environment the paper's pinned numbers depend on.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python --version
python - <<'PY'
import importlib.metadata as m

for pkg in ("agentdojo", "pytest", "ruff", "mypy", "PyYAML"):
    try:
        print(f"{pkg} {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"{pkg} NOT INSTALLED")
PY
uname -sm
