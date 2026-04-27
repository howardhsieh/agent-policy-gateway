"""Smoke test: package imports and version is exposed."""

import agent_policy_gateway as apg


def test_version_present() -> None:
    assert isinstance(apg.__version__, str)
    assert apg.__version__.count(".") == 2
