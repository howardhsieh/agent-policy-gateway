"""Declarative policy DSL for agent-policy-gateway (R3).

A *policy file* is a YAML document with a small, fixed schema::

    version: 1                       # int — currently must be 1
    name: my-policy                  # str — required
    description: ...                 # optional human-readable description
    rules:
      - id: deny-web-to-email        # str — required, unique within policy
        description: ...             # optional
        when:                        # selector — every field optional
          tool: send_email           # str or fnmatch glob, optional
          identity: agent.research   # str, optional (matches ToolCall.agent_id)
          resource: "https://*"      # str or fnmatch glob, optional
          taint:                     # optional condition on input taint
            any_of: [web]            # at least one of these sources present
            all_of: []               # all of these sources present
            none_of: []              # none of these sources present
        effect:
          action: deny               # allow | deny | review | rate_limit
          reason: "..."              # optional
          limit_per_minute: 30       # required iff action == rate_limit

The loader (:func:`load_policy`) parses YAML, validates with Pydantic, and
returns a frozen :class:`Policy` object. The validator rejects unknown
fields and enforces effect-shape invariants (e.g. ``rate_limit`` requires
a positive ``limit_per_minute``; other actions must omit it).

This is the *static* half of policy enforcement. The reference monitor in
``gateway.py`` (R4) will walk ``policy.rules`` in order and apply the
first matching rule's effect — but that lives in the next roadmap item.
This module is intentionally pure: no I/O beyond reading the YAML file
and no mutation of any runtime state.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agent_policy_gateway.core import TaintLabel, ToolCall


class PolicyError(ValueError):
    """Raised when a policy file is structurally invalid."""


class Action(str, Enum):
    """The four effect actions a rule may take."""

    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"
    RATE_LIMIT = "rate_limit"


class TaintCondition(BaseModel):
    """Boolean condition on the input taint label of a tool call.

    A condition is satisfied iff *all three* clauses (``all_of``,
    ``any_of``, ``none_of``) are individually satisfied. An unset clause
    is trivially true. An empty :class:`TaintCondition` (no clauses at
    all) matches every label.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    any_of: tuple[str, ...] = ()
    all_of: tuple[str, ...] = ()
    none_of: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        """True iff this condition has no clauses (matches every label)."""
        return not (self.any_of or self.all_of or self.none_of)

    def matches(self, label: TaintLabel) -> bool:
        """True iff ``label`` satisfies every non-empty clause."""
        srcs = label.sources
        if self.all_of and not set(self.all_of).issubset(srcs):
            return False
        if self.any_of and not (set(self.any_of) & srcs):
            return False
        if self.none_of and (set(self.none_of) & srcs):
            return False
        return True


class Selector(BaseModel):
    """Match conditions on a :class:`ToolCall`.

    Every field is optional. A field that is ``None`` does not constrain
    the match. An empty selector (all fields ``None``) matches every
    call. ``tool`` and ``resource`` use :mod:`fnmatch`-style globbing
    so policies can match families of tools (e.g. ``send_*``) or URL
    prefixes (e.g. ``https://*``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str | None = None
    identity: str | None = None
    resource: str | None = None
    taint: TaintCondition | None = None

    def matches(self, call: ToolCall, *, resource: str | None = None) -> bool:
        """Return True iff this selector matches ``call``.

        ``resource`` is supplied by the caller (typically the gateway)
        when the tool exposes a target resource that should be matched
        against ``Selector.resource``. If the selector has a ``resource``
        glob but the caller passed ``resource=None``, the selector does
        not match (a resource constraint cannot be satisfied without a
        resource to inspect).
        """
        if self.tool is not None and not fnmatch.fnmatchcase(call.tool_name, self.tool):
            return False
        if self.identity is not None and call.agent_id != self.identity:
            return False
        if self.resource is not None:
            if resource is None or not fnmatch.fnmatchcase(resource, self.resource):
                return False
        if self.taint is not None and not self.taint.matches(call.input_label):
            return False
        return True


class Effect(BaseModel):
    """The action a matched rule applies to a tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    reason: str = ""
    limit_per_minute: int | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> Effect:
        if self.action == Action.RATE_LIMIT:
            if self.limit_per_minute is None or self.limit_per_minute <= 0:
                raise ValueError(
                    "rate_limit effect requires a positive limit_per_minute"
                )
        else:
            if self.limit_per_minute is not None:
                raise ValueError(
                    "limit_per_minute is only allowed for action=rate_limit "
                    f"(got action={self.action.value})"
                )
        return self


class Rule(BaseModel):
    """A single rule: selector + effect, with a stable identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    description: str = ""
    when: Selector = Field(default_factory=Selector)
    effect: Effect

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rule id must be a non-empty string")
        return v


class Policy(BaseModel):
    """A named, versioned ordered list of rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    name: str
    description: str = ""
    rules: tuple[Rule, ...] = ()

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("policy name must be a non-empty string")
        return v

    @field_validator("version")
    @classmethod
    def _supported_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"unsupported policy version: {v} (only v1 is supported)")
        return v

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> Policy:
        seen: set[str] = set()
        for r in self.rules:
            if r.id in seen:
                raise ValueError(f"duplicate rule id: {r.id!r}")
            seen.add(r.id)
        return self

    def first_match(
        self,
        call: ToolCall,
        *,
        resource: str | None = None,
    ) -> Rule | None:
        """Return the first rule whose selector matches ``call``, else ``None``."""
        for rule in self.rules:
            if rule.when.matches(call, resource=resource):
                return rule
        return None


def load_policy_str(text: str, *, source: str = "<string>") -> Policy:
    """Parse and validate a policy from a YAML string.

    ``source`` is used purely to make error messages readable and is
    safe to omit for ad-hoc strings.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PolicyError(f"{source}: invalid YAML: {e}") from e
    if data is None:
        raise PolicyError(f"{source}: policy file is empty")
    if not isinstance(data, dict):
        raise PolicyError(
            f"{source}: top-level YAML must be a mapping, got {type(data).__name__}"
        )
    try:
        return Policy.model_validate(data)
    except ValidationError as e:
        raise PolicyError(f"{source}: {e}") from e


def load_policy(source: str | os.PathLike[str] | Path) -> Policy:
    """Read, parse and validate a policy from a YAML file path."""
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    return load_policy_str(text, source=str(path))


def load_policies(paths: Iterable[str | os.PathLike[str]]) -> list[Policy]:
    """Convenience: load multiple policies from a list of paths."""
    return [load_policy(p) for p in paths]


__all__ = [
    "Action",
    "Effect",
    "Policy",
    "PolicyError",
    "Rule",
    "Selector",
    "TaintCondition",
    "load_policies",
    "load_policy",
    "load_policy_str",
]
