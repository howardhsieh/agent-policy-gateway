"""agent-policy-gateway: policy enforcement and IFC for AI agent tool calls."""

from agent_policy_gateway.core import (
    Decision,
    TaintLabel,
    ToolCall,
    Verdict,
    from_json,
    to_json,
)
from agent_policy_gateway.gateway import (
    AGENT_ID_KWARG,
    CALL_ID_KWARG,
    INPUT_LABEL_KWARG,
    RESOURCE_KWARG,
    AuditWriter,
    Gateway,
    GatewayError,
    PolicyDenied,
    PolicyReview,
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
    "AGENT_ID_KWARG",
    "Action",
    "AuditWriter",
    "CALL_ID_KWARG",
    "Decision",
    "Effect",
    "Gateway",
    "GatewayError",
    "INPUT_LABEL_KWARG",
    "Policy",
    "PolicyDenied",
    "PolicyError",
    "PolicyReview",
    "RESOURCE_KWARG",
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
