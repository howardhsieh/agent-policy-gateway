"""agent-policy-gateway: policy enforcement and IFC for AI agent tool calls."""

from agent_policy_gateway.core import (
    Decision,
    TaintLabel,
    ToolCall,
    Verdict,
    from_json,
    to_json,
)
from agent_policy_gateway.policy import (
    Action,
    Effect,
    Policy,
    PolicyError,
    Rule,
    Selector,
    TaintCondition,
    load_policies,
    load_policy,
    load_policy_str,
)
from agent_policy_gateway.taint import (
    ToolTaintSpec,
    flows_to,
    join,
    join_all,
    propagate,
    subsumes,
)

__version__ = "0.0.1"

__all__ = [
    "Action",
    "Decision",
    "Effect",
    "Policy",
    "PolicyError",
    "Rule",
    "Selector",
    "TaintCondition",
    "TaintLabel",
    "ToolCall",
    "ToolTaintSpec",
    "Verdict",
    "__version__",
    "flows_to",
    "from_json",
    "join",
    "join_all",
    "load_policies",
    "load_policy",
    "load_policy_str",
    "propagate",
    "subsumes",
    "to_json",
]
