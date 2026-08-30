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
          arg_equals:                # optional literal argument-value matching
            channel: "#public"       # str / int / bool, compared by equality
          arg_matches:               # optional regex argument matching (R54)
            recipient: "^UK\\d+$"    # re.search over string values only
          taint:                     # optional condition on input taint
            any_of: [web]            # at least one of these sources present
            all_of: []               # all of these sources present
            none_of: []              # none of these sources present
            confidentiality:         # optional per-dimension clauses (R51):
              any_of: [pii]          #   matched against the confidentiality
            integrity:               #   (resp. integrity) effective set
              none_of: [web]
          chain:                     # optional chain-level condition (R53)
            any_prior:               # >=1 recorded prior call matches >=1 matcher
              - tool: "get_*"        #   fnmatch glob over the prior call's tool
                source: web          #   glob over its output label's sources
                verdict: allow       #   exact verdict; unset = any (denials too)
                resource: "https://*"
            no_prior: []             # no recorded prior call matches any matcher
            provenance:              # condition on the input's provenance chain
              any_of:
                - {source: web, tool: "get_*"}
              none_of: []
        effect:
          action: deny               # allow | deny | review | rate_limit
          reason: "..."              # optional
          limit_per_minute: 30       # required iff action == rate_limit
    declassify:                      # optional declassification grants (R52)
      - id: sanitizer-endorses-web   # str — required, unique among grants
        tool: sanitize_html          # str or fnmatch glob — required
        identity: agent.research     # optional (matches ToolCall.agent_id)
        resource: "https://*"        # optional fnmatch glob
        sources: [web, "rss.*"]      # fnmatch globs over source names — required
        dimensions: [integrity]      # subset of {confidentiality, integrity};
                                     #   default: both
        when:                        # optional condition on input taint
          none_of: [pii]

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
import re
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from agent_policy_gateway.core import (
    CallHistoryEntry,
    Provenance,
    ProvenanceEntry,
    TaintLabel,
    ToolCall,
    Verdict,
)


class PolicyError(ValueError):
    """Raised when a policy file is structurally invalid."""


class Action(str, Enum):
    """The four effect actions a rule may take."""

    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"
    RATE_LIMIT = "rate_limit"
    REDACT = "redact"


def _clauses_match(
    srcs: frozenset[str],
    *,
    any_of: tuple[str, ...],
    all_of: tuple[str, ...],
    none_of: tuple[str, ...],
) -> bool:
    """Shared any_of/all_of/none_of matching against a source set."""
    if all_of and not set(all_of).issubset(srcs):
        return False
    if any_of and not (set(any_of) & srcs):
        return False
    if none_of and (set(none_of) & srcs):
        return False
    return True


class DimensionTaintCondition(BaseModel):
    """Boolean condition on one dimension of a dual-dimension label (R51).

    Matched against that dimension's *effective* source set (legacy
    ``sources`` count in both dimensions). An empty condition matches
    every label.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    any_of: tuple[str, ...] = ()
    all_of: tuple[str, ...] = ()
    none_of: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        """True iff this condition has no clauses (matches every label)."""
        return not (self.any_of or self.all_of or self.none_of)

    def matches_sources(self, srcs: frozenset[str]) -> bool:
        """True iff the dimension's effective set satisfies every clause."""
        return _clauses_match(
            srcs, any_of=self.any_of, all_of=self.all_of, none_of=self.none_of
        )


class TaintCondition(BaseModel):
    """Boolean condition on the input taint label of a tool call.

    The top-level ``any_of`` / ``all_of`` / ``none_of`` clauses are
    matched against the **union** of the label's dimensions — for legacy
    single-set labels that is exactly the pre-R51 behavior. The nested
    ``confidentiality:`` / ``integrity:`` sub-conditions (R51) are each
    matched against that dimension's effective set, so a policy can
    require e.g. "no untrusted integrity taint" without caring whether
    the same sources also count as secret.

    A condition is satisfied iff *all* its clauses (top-level and
    per-dimension) are individually satisfied. An unset clause is
    trivially true. An empty :class:`TaintCondition` (no clauses at all)
    matches every label.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    any_of: tuple[str, ...] = ()
    all_of: tuple[str, ...] = ()
    none_of: tuple[str, ...] = ()
    confidentiality: DimensionTaintCondition | None = None
    integrity: DimensionTaintCondition | None = None

    def is_empty(self) -> bool:
        """True iff this condition has no clauses (matches every label)."""
        return not (
            self.any_of
            or self.all_of
            or self.none_of
            or (self.confidentiality is not None and not self.confidentiality.is_empty())
            or (self.integrity is not None and not self.integrity.is_empty())
        )

    def matches(self, label: TaintLabel) -> bool:
        """True iff ``label`` satisfies every non-empty clause."""
        if not _clauses_match(
            label.all_sources,
            any_of=self.any_of,
            all_of=self.all_of,
            none_of=self.none_of,
        ):
            return False
        if self.confidentiality is not None and not self.confidentiality.matches_sources(
            label.confidentiality_sources
        ):
            return False
        if self.integrity is not None and not self.integrity.matches_sources(
            label.integrity_sources
        ):
            return False
        return True


class PriorCallMatcher(BaseModel):
    """Matches one recorded call in the session history (R53).

    Every field is optional and glob-valued (except ``verdict``); an
    entry matches when *all* set fields match. An empty matcher matches
    any recorded call, so ``any_prior: [{}]`` reads "any prior call at
    all".

    Attributes:
        tool: fnmatch glob over the prior call's tool name.
        source: fnmatch glob matched against the prior call's *output
            label* — satisfied when any source in any dimension matches,
            so ``source: web`` reads "a call whose output carried web
            taint".
        verdict: exact verdict the prior call received (``allow`` /
            ``deny`` / ``review`` / ``redact``). Unset matches any —
            note history records denied *attempts* too; set
            ``verdict: allow`` when only executed calls should count.
        resource: fnmatch glob over the prior call's resource. A prior
            call recorded with no resource never matches a set glob
            (the ``Selector.resource`` precedent).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str | None = None
    source: str | None = None
    verdict: Verdict | None = None
    resource: str | None = None

    def matches_entry(self, entry: CallHistoryEntry) -> bool:
        """True iff ``entry`` satisfies every set field."""
        if self.tool is not None and not fnmatch.fnmatchcase(
            entry.tool_name, self.tool
        ):
            return False
        if self.verdict is not None and entry.verdict != self.verdict:
            return False
        if self.source is not None and not any(
            fnmatch.fnmatchcase(s, self.source)
            for s in entry.output_label.all_sources
        ):
            return False
        if self.resource is not None:
            if entry.resource is None or not fnmatch.fnmatchcase(
                entry.resource, self.resource
            ):
                return False
        return True


class ProvenanceMatcher(BaseModel):
    """Matches one entry of the input's provenance chain (R53).

    ``source`` globs the taint source the entry records; ``tool`` globs
    the tool that introduced it. An entry matches when all set fields
    match; an empty matcher matches any entry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str | None = None
    tool: str | None = None

    def matches_entry(self, entry: ProvenanceEntry) -> bool:
        """True iff ``entry`` satisfies every set field."""
        if self.source is not None and not fnmatch.fnmatchcase(
            entry.source, self.source
        ):
            return False
        if self.tool is not None and not fnmatch.fnmatchcase(
            entry.tool_name, self.tool
        ):
            return False
        return True


class ProvenanceCondition(BaseModel):
    """Boolean condition on the input's provenance chain (R53).

    ``any_of`` is satisfied when some chain entry matches some matcher;
    ``none_of`` when no chain entry matches any matcher. Provenance
    chains are populated only by a gateway with
    ``track_provenance=True`` — against an empty chain, ``any_of``
    cannot be satisfied and ``none_of`` is trivially satisfied.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    any_of: tuple[ProvenanceMatcher, ...] = ()
    none_of: tuple[ProvenanceMatcher, ...] = ()

    def is_empty(self) -> bool:
        """True iff this condition has no clauses (matches everything)."""
        return not (self.any_of or self.none_of)

    def matches(self, provenance: Provenance) -> bool:
        """True iff ``provenance`` satisfies every non-empty clause."""
        if self.any_of and not any(
            m.matches_entry(e) for e in provenance.entries for m in self.any_of
        ):
            return False
        if self.none_of and any(
            m.matches_entry(e) for e in provenance.entries for m in self.none_of
        ):
            return False
        return True


class ChainCondition(BaseModel):
    """Chain-level condition over call history and provenance (R53).

    ``any_prior`` is satisfied when at least one recorded prior call in
    the session matches at least one of its matchers; ``no_prior`` when
    no recorded prior call matches any of its matchers. ``provenance``
    constrains the *current call's input* provenance chain.

    The history clauses need a history to inspect: against a gateway
    that does not track history (``history=None``, as opposed to an
    empty recorded history) neither clause can be verified, so a
    selector carrying one does not match — the ``Selector.resource``
    precedent, and the reason chain policies require a gateway with
    ``track_history=True``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    any_prior: tuple[PriorCallMatcher, ...] = ()
    no_prior: tuple[PriorCallMatcher, ...] = ()
    provenance: ProvenanceCondition | None = None

    def is_empty(self) -> bool:
        """True iff this condition has no clauses (matches every call)."""
        return not (
            self.any_prior
            or self.no_prior
            or (self.provenance is not None and not self.provenance.is_empty())
        )

    def needs_history(self) -> bool:
        """True iff this condition references the session call history."""
        return bool(self.any_prior or self.no_prior)

    def matches(
        self,
        call: ToolCall,
        *,
        history: Sequence[CallHistoryEntry] | None = None,
    ) -> bool:
        """True iff ``call`` (with its session ``history``) satisfies
        every non-empty clause."""
        if self.needs_history():
            if history is None:
                return False
            if self.any_prior and not any(
                m.matches_entry(e) for e in history for m in self.any_prior
            ):
                return False
            if self.no_prior and any(
                m.matches_entry(e) for e in history for m in self.no_prior
            ):
                return False
        if self.provenance is not None and not self.provenance.matches(
            call.input_provenance
        ):
            return False
        return True


def _arg_value_equal(expected: object, actual: object) -> bool:
    """Equality for ``arg_equals`` with bool/int type strictness.

    Python treats ``True == 1`` as true, but YAML's ``true`` and ``1`` are
    distinct scalar types — a policy author who writes ``flag: true``
    should not match a call passing ``flag=1`` (or vice versa).
    """
    if isinstance(expected, bool) != isinstance(actual, bool):
        return False
    return expected == actual


class Selector(BaseModel):
    """Match conditions on a :class:`ToolCall`.

    Every field is optional. A field that is ``None`` does not constrain
    the match. An empty selector (all fields ``None``) matches every
    call. ``tool`` and ``resource`` use :mod:`fnmatch`-style globbing
    so policies can match families of tools (e.g. ``send_*``) or URL
    prefixes (e.g. ``https://*``). ``arg_equals`` matches named call
    arguments against literal scalar values (``str`` / ``int`` /
    ``bool``): every listed argument must be present on the call and
    equal to the given value. Comparison is type-strict between ``bool``
    and ``int`` (``true`` does not match ``1``). An empty ``arg_equals``
    mapping, like an absent one, does not constrain the match.

    ``arg_matches`` (R54) matches named call arguments against regular
    expressions with :func:`re.search` semantics — the JSON Schema
    ``pattern`` convention, so imported Progent rules keep their
    meaning. Every listed argument must be present on the call, be a
    string, and contain a match; a non-string value never matches
    (anchor with ``\\A...\\Z`` for full-value matching). The empty
    pattern therefore reads "any string". Like ``arg_equals``, an empty
    mapping does not constrain the match, and both may constrain the
    same argument (the call must satisfy each independently).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool: str | None = None
    identity: str | None = None
    resource: str | None = None
    arg_equals: dict[str, StrictStr | StrictInt | StrictBool] | None = None
    arg_matches: dict[str, StrictStr] | None = None
    taint: TaintCondition | None = None
    chain: ChainCondition | None = None

    @field_validator("arg_equals")
    @classmethod
    def _arg_keys_nonempty(
        cls, v: dict[str, StrictStr | StrictInt | StrictBool] | None
    ) -> dict[str, StrictStr | StrictInt | StrictBool] | None:
        if v is not None and any(not k.strip() for k in v):
            raise ValueError("arg_equals keys must be non-empty argument names")
        return v

    @field_validator("arg_matches")
    @classmethod
    def _arg_patterns_valid(
        cls, v: dict[str, StrictStr] | None
    ) -> dict[str, StrictStr] | None:
        if v is None:
            return v
        if any(not k.strip() for k in v):
            raise ValueError("arg_matches keys must be non-empty argument names")
        for key, pattern in v.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(
                    f"arg_matches[{key!r}] is not a valid regex: {e}"
                ) from e
        return v

    def matches(
        self,
        call: ToolCall,
        *,
        resource: str | None = None,
        history: Sequence[CallHistoryEntry] | None = None,
    ) -> bool:
        """Return True iff this selector matches ``call``.

        ``resource`` is supplied by the caller (typically the gateway)
        when the tool exposes a target resource that should be matched
        against ``Selector.resource``. If the selector has a ``resource``
        glob but the caller passed ``resource=None``, the selector does
        not match (a resource constraint cannot be satisfied without a
        resource to inspect).

        ``history`` is the session's recorded call history (R53),
        supplied by a gateway with ``track_history=True``. ``None``
        means *no history is tracked* — a chain condition referencing
        prior calls then does not match, exactly like the unsatisfiable
        resource constraint — while an empty sequence means "no prior
        calls recorded", which ``no_prior:`` clauses can satisfy.
        """
        if self.tool is not None and not fnmatch.fnmatchcase(call.tool_name, self.tool):
            return False
        if self.identity is not None and call.agent_id != self.identity:
            return False
        if self.resource is not None:
            if resource is None or not fnmatch.fnmatchcase(resource, self.resource):
                return False
        if self.arg_equals:
            for key, expected in self.arg_equals.items():
                if key not in call.args:
                    return False
                if not _arg_value_equal(expected, call.args[key]):
                    return False
        if self.arg_matches:
            for key, pattern in self.arg_matches.items():
                if key not in call.args:
                    return False
                value = call.args[key]
                if not isinstance(value, str) or re.search(pattern, value) is None:
                    return False
        if self.taint is not None and not self.taint.matches(call.input_label):
            return False
        if self.chain is not None and not self.chain.matches(call, history=history):
            return False
        return True


class RedactSpec(BaseModel):
    """How a ``redact`` effect transforms a tool call's arguments.

    A redact effect masks one or more argument *fields* in place and lets
    the call proceed with a *declassified* output label. It is the middle
    ground between ``allow`` (let tainted data through untouched) and
    ``deny`` (refuse the call): the sensitive substring is removed and the
    now-clean value is forwarded to the downstream tool.

    Attributes:
        fields: Argument names to transform. Required and non-empty.
        pattern: Optional regular expression. When set, only the
            substrings matching ``pattern`` inside each field are replaced
            with ``mask`` (so surrounding text is preserved). When unset,
            the entire field value is replaced with ``mask``.
        mask: Replacement text. Defaults to ``"[REDACTED]"``.
        declassify: Taint sources stripped from the output label once the
            fields are masked (e.g. ``pii``).
        add_label: Taint sources added to the output label to mark that a
            vetted redaction ran (e.g. ``trusted-redactor``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fields: tuple[str, ...]
    pattern: str | None = None
    mask: str = "[REDACTED]"
    declassify: tuple[str, ...] = ()
    add_label: tuple[str, ...] = ()

    @field_validator("fields")
    @classmethod
    def _fields_nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("redact.fields must list at least one field name")
        if any(not f.strip() for f in v):
            raise ValueError("redact.fields entries must be non-empty strings")
        return v

    @field_validator("pattern")
    @classmethod
    def _pattern_compiles(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"redact.pattern is not a valid regex: {e}") from e
        return v

    def compiled_pattern(self) -> re.Pattern[str] | None:
        """Return the compiled ``pattern`` regex, or ``None`` when unset."""
        return re.compile(self.pattern) if self.pattern is not None else None

    def redact_value(self, value: object) -> object:
        """Return the masked form of a single field ``value``.

        With a ``pattern`` set, string values have matching substrings
        replaced and non-string values are returned untouched (a regex
        cannot meaningfully match them). With no ``pattern`` the whole
        value is replaced by ``mask`` regardless of type.
        """
        pat = self.compiled_pattern()
        if pat is not None:
            if isinstance(value, str):
                return pat.sub(self.mask, value)
            return value
        return self.mask


#: The two label dimensions a declassify grant may act on (R51 vocabulary).
DIMENSIONS: tuple[str, ...] = ("confidentiality", "integrity")


class DeclassifyGrant(BaseModel):
    """A policy-declared declassification authority (R52).

    A grant names a *tool* (fnmatch glob) that is permitted to strip the
    listed ``sources`` (fnmatch globs over concrete source names) from
    the listed ``dimensions`` of its output label, optionally conditioned
    on the caller ``identity``, the target ``resource``, and the input
    taint (``when:``). Stripping from the confidentiality dimension is
    declassification proper; stripping from the integrity dimension is
    *endorsement* — the R51 vocabulary.

    When any loaded policy carries grants, the gateway treats the policy
    as the sole authority on declassification: per-spec
    ``ToolTaintSpec.declassifies`` is inert and only grants strip (see
    ``docs/design.md``, "Declarative declassify (R52)").
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    description: str = ""
    tool: str
    identity: str | None = None
    resource: str | None = None
    sources: tuple[str, ...]
    dimensions: tuple[str, ...] = DIMENSIONS
    when: TaintCondition | None = None

    @field_validator("id", "tool")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("declassify grant id and tool must be non-empty")
        return v

    @field_validator("sources")
    @classmethod
    def _sources_nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("declassify.sources must list at least one source")
        if any(not s.strip() for s in v):
            raise ValueError("declassify.sources entries must be non-empty strings")
        return v

    @field_validator("dimensions")
    @classmethod
    def _known_dimensions(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("declassify.dimensions must list at least one dimension")
        for d in v:
            if d not in DIMENSIONS:
                raise ValueError(
                    f"unknown declassify dimension {d!r} "
                    f"(expected one of {list(DIMENSIONS)})"
                )
        if len(set(v)) != len(v):
            raise ValueError("declassify.dimensions entries must be unique")
        return v

    def matches(self, call: ToolCall, *, resource: str | None = None) -> bool:
        """True iff this grant's conditions are satisfied by ``call``.

        Mirrors :meth:`Selector.matches` semantics field for field: a
        ``resource`` constraint with no runtime resource to inspect does
        not match, and ``when:`` is evaluated against the call's input
        label.
        """
        if not fnmatch.fnmatchcase(call.tool_name, self.tool):
            return False
        if self.identity is not None and call.agent_id != self.identity:
            return False
        if self.resource is not None:
            if resource is None or not fnmatch.fnmatchcase(resource, self.resource):
                return False
        if self.when is not None and not self.when.matches(call.input_label):
            return False
        return True

    def strips(self, srcs: frozenset[str], dimension: str) -> frozenset[str]:
        """The subset of concrete ``srcs`` this grant strips from ``dimension``.

        Empty when the grant does not act on ``dimension``; otherwise every
        source matching any of the grant's ``sources`` globs.
        """
        if dimension not in self.dimensions:
            return frozenset()
        return frozenset(
            s
            for s in srcs
            if any(fnmatch.fnmatchcase(s, pat) for pat in self.sources)
        )


class Effect(BaseModel):
    """The action a matched rule applies to a tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    reason: str = ""
    limit_per_minute: int | None = None
    redact: RedactSpec | None = None

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
        if self.action == Action.REDACT:
            if self.redact is None:
                raise ValueError("redact effect requires a redact block")
        elif self.redact is not None:
            raise ValueError(
                "redact is only allowed for action=redact "
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
    declassify: tuple[DeclassifyGrant, ...] = ()

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
        grant_seen: set[str] = set()
        for g in self.declassify:
            if g.id in grant_seen:
                raise ValueError(f"duplicate declassify grant id: {g.id!r}")
            grant_seen.add(g.id)
        return self

    def matching_grants(
        self,
        call: ToolCall,
        *,
        resource: str | None = None,
    ) -> tuple[DeclassifyGrant, ...]:
        """Every declassify grant (R52) whose conditions match ``call``.

        Unlike rule matching this is not first-match: all matching grants
        contribute (their strips union), in declaration order.
        """
        return tuple(
            g for g in self.declassify if g.matches(call, resource=resource)
        )

    def first_match(
        self,
        call: ToolCall,
        *,
        resource: str | None = None,
        history: Sequence[CallHistoryEntry] | None = None,
    ) -> Rule | None:
        """Return the first rule whose selector matches ``call``, else ``None``.

        ``history`` is the session call history for chain-level rules
        (R53); ``None`` — the default, and what a gateway without
        ``track_history`` passes — means history-referencing chain
        conditions never match.
        """
        for rule in self.rules:
            if rule.when.matches(call, resource=resource, history=history):
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
    "ChainCondition",
    "DIMENSIONS",
    "DeclassifyGrant",
    "DimensionTaintCondition",
    "Effect",
    "Policy",
    "PolicyError",
    "PriorCallMatcher",
    "ProvenanceCondition",
    "ProvenanceMatcher",
    "RedactSpec",
    "Rule",
    "Selector",
    "TaintCondition",
    "load_policies",
    "load_policy",
    "load_policy_str",
]
