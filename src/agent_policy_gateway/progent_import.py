"""Progent symbolic-rule import PoC (R54).

Progent (Shi et al., "Progent: Programmable Privilege Control for LLM
Agents"; ``sunblaze-ucb/progent``) enforces per-tool privilege policies
whose runtime representation is a mapping::

    {tool_name: [(priority, effect, condition, fallback), ...]}

verified against ``secagent/tool.py`` at upstream ``main``:

* ``priority`` — int; rules are sorted by ``(priority, -effect)``, so
  lower priority numbers run first and, at equal priority, *forbid*
  rules run before *allow* rules.
* ``effect`` — ``0`` = allow, ``1`` = forbid.
* ``condition`` — a dict mapping argument names to *restrictions*: a
  JSON Schema fragment (checked with ``jsonschema.validate``), a bare
  regex string (checked with ``re.match``), or a Python callable.
* ``fallback`` — what a firing forbid rule does: ``0`` = raise an error
  message back to the agent, ``1`` = terminate the agent process,
  ``2`` = ask the user for confirmation.
* A tool with no entry is denied, and a call that satisfies no rule of
  its tool falls off the end and is denied (with the quirk that the
  *last examined rule's* fallback decides how — see below).
* An *allow* rule at ``priority == 100`` with ``fallback == 0`` is
  "hard": an argument that is present but fails its restriction denies
  the call immediately, before any later rule is consulted.

:func:`load_progent_policy` reads that mapping from JSON (the natural
serialization of the structure above — Python callables are not
representable and are rejected), and :func:`convert_progent_policy`
translates it into an ordered first-match APG :class:`Policy`:

===========================================  ================================
Progent construct                            APG translation
===========================================  ================================
sorted rule order ``(priority, -effect)``    first-match rule order
allow rule (effect 0)                        ``action: allow`` rule(s)
forbid rule, fallback 0                      ``action: deny`` rule(s)
forbid rule, fallback 1 (terminate)          ``action: deny`` (reason notes
                                             the terminate divergence — APG
                                             never kills the process)
forbid rule, fallback 2 (user confirm)       ``action: review`` rule(s)
bare-string regex restriction (re.match)     ``arg_matches`` with the
                                             pattern anchored as ``\\A(?:…)``
JSON Schema ``pattern``                      ``arg_matches`` (both are
                                             ``re.search`` semantics)
JSON Schema ``const`` / ``enum``             ``arg_equals``, one rule per
                                             value combination (capped)
JSON Schema ``{"type": "string"}``           ``arg_matches`` with the empty
                                             pattern ("any string")
hard allow (priority 100, fallback 0)        the allow rule(s) followed by a
                                             tool-wide ``deny``
fall off the end of a tool's rules           tool-wide trailing rule whose
                                             action maps the *last* rule's
                                             fallback (0/1 → deny, 2 →
                                             review) — Progent's leaky-loop-
                                             variable quirk, kept faithfully
tool with no entry                           trailing catch-all ``deny``
                                             (``default="deny"``)
===========================================  ================================

Anything outside the subset — callables, five-element self-updating
rules (``need_update_policies``), JSON Schema keywords other than
``const`` / ``enum`` / ``pattern`` / ``type: string``, float or null
scalars — raises :class:`ProgentImportError`. The import is never
silently weaker than the source policy.

Documented divergences (see ``docs/design.md``, "Progent rule import
(R54)"):

* **Absent arguments.** Progent checks a restriction only when the
  argument is present in the call — an allow rule's restriction on an
  absent argument passes vacuously, and a forbid rule's restriction on
  an absent argument counts as matched. APG's ``arg_equals`` /
  ``arg_matches`` require the argument to be present, so translated
  allow rules are *stricter* (a call omitting a constrained argument is
  not allowed by that rule) and translated forbid rules do not fire on
  absent arguments (the call then falls through to the trailing
  default, which denies unless a later allow rule admits it).
* **Non-string values under ``pattern``.** JSON Schema treats string
  keywords as inapplicable to non-string instances (vacuously valid);
  APG's ``arg_matches`` never matches a non-string value. Fail-closed
  for allow rules.
* **Terminate.** ``fallback == 1`` maps to ``deny``; APG never calls
  ``sys.exit``.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

from agent_policy_gateway.policy import (
    Action,
    Effect,
    Policy,
    Rule,
    Selector,
)

#: Progent effect codes.
PROGENT_ALLOW = 0
PROGENT_FORBID = 1

#: Progent fallback codes.
FALLBACK_ERROR = 0
FALLBACK_TERMINATE = 1
FALLBACK_CONFIRM = 2

#: The priority Progent treats as "hard" for allow rules: a present but
#: failing argument denies immediately (``fallback == 0`` only).
HARD_PRIORITY = 100

#: Cap on the ``enum`` cross-product expansion of a single Progent rule.
MAX_ENUM_EXPANSION = 64

#: A Progent argument restriction as it appears in JSON: a bare regex
#: string or a JSON Schema fragment. (Upstream also accepts callables,
#: which JSON cannot carry — the loader rejects anything else.)
Restriction = str | dict

#: The JSON Schema keywords the translator understands.
_SUPPORTED_KEYWORDS = frozenset({"const", "enum", "pattern", "type"})

#: Literal scalars ``arg_equals`` can carry.
_Scalar = str | int | bool


class ProgentImportError(ValueError):
    """Raised when a Progent policy is malformed or outside the subset."""


@dataclass(frozen=True)
class ProgentRule:
    """One Progent rule: ``(priority, effect, condition, fallback)``."""

    priority: int
    effect: int
    condition: dict[str, Restriction]
    fallback: int


def progent_sorted(rules: list[ProgentRule]) -> list[ProgentRule]:
    """Progent's evaluation order: ``sorted(key=(priority, -effect))``.

    Lower priority numbers first; at equal priority, forbid (effect 1)
    before allow (effect 0) — verbatim from upstream ``sort_policy``.
    """
    return sorted(rules, key=lambda r: (r.priority, -r.effect))


def _parse_rule(tool: str, index: int, item: object) -> ProgentRule:
    where = f"tool {tool!r} rule #{index}"
    if not isinstance(item, (list, tuple)):
        raise ProgentImportError(
            f"{where}: expected a [priority, effect, condition, fallback] "
            f"array, got {type(item).__name__}"
        )
    if len(item) == 5:
        raise ProgentImportError(
            f"{where}: self-updating rules (need_update_policies) are not "
            "supported by the import"
        )
    if len(item) != 4:
        raise ProgentImportError(
            f"{where}: expected 4 elements (priority, effect, condition, "
            f"fallback), got {len(item)}"
        )
    priority, effect, condition, fallback = item
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ProgentImportError(f"{where}: priority must be an int")
    if effect not in (PROGENT_ALLOW, PROGENT_FORBID) or isinstance(effect, bool):
        raise ProgentImportError(
            f"{where}: effect must be 0 (allow) or 1 (forbid), got {effect!r}"
        )
    if not isinstance(condition, dict):
        raise ProgentImportError(
            f"{where}: condition must be a dict of argument restrictions, "
            f"got {type(condition).__name__}"
        )
    for arg, restriction in condition.items():
        if not isinstance(arg, str) or not arg.strip():
            raise ProgentImportError(
                f"{where}: condition keys must be non-empty argument names"
            )
        if not isinstance(restriction, (str, dict)):
            raise ProgentImportError(
                f"{where}: restriction for argument {arg!r} must be a JSON "
                "Schema dict or a regex string (callables are not "
                f"JSON-representable), got {type(restriction).__name__}"
            )
    if fallback not in (FALLBACK_ERROR, FALLBACK_TERMINATE, FALLBACK_CONFIRM) or isinstance(
        fallback, bool
    ):
        raise ProgentImportError(
            f"{where}: fallback must be 0 (error message), 1 (terminate) "
            f"or 2 (user confirm), got {fallback!r}"
        )
    return ProgentRule(
        priority=priority, effect=effect, condition=dict(condition), fallback=fallback
    )


def parse_progent_policy(data: object) -> dict[str, list[ProgentRule]]:
    """Validate a decoded Progent policy mapping into :class:`ProgentRule` lists."""
    if not isinstance(data, dict):
        raise ProgentImportError(
            "top-level Progent policy must be a mapping of tool name to "
            f"rule list, got {type(data).__name__}"
        )
    result: dict[str, list[ProgentRule]] = {}
    for tool, rules in data.items():
        if not isinstance(tool, str) or not tool.strip():
            raise ProgentImportError("tool names must be non-empty strings")
        if not isinstance(rules, list):
            raise ProgentImportError(
                f"tool {tool!r}: expected a list of rules, got "
                f"{type(rules).__name__}"
            )
        result[tool] = [_parse_rule(tool, i, item) for i, item in enumerate(rules)]
    return result


def load_progent_policy_str(
    text: str, *, source: str = "<string>"
) -> dict[str, list[ProgentRule]]:
    """Parse and validate a Progent policy from a JSON string."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProgentImportError(f"{source}: invalid JSON: {e}") from e
    try:
        return parse_progent_policy(data)
    except ProgentImportError as e:
        raise ProgentImportError(f"{source}: {e}") from e


def load_progent_policy(
    source: str | os.PathLike[str] | Path,
) -> dict[str, list[ProgentRule]]:
    """Read, parse and validate a Progent policy from a JSON file path."""
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    return load_progent_policy_str(text, source=str(path))


def _glob_escape(name: str) -> str:
    """Escape ``name`` so ``fnmatch`` treats it as a literal tool name."""
    return re.sub(r"([*?[])", r"[\1]", name)


def _compile(where: str, arg: str, pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as e:
        raise ProgentImportError(
            f"{where}: restriction for argument {arg!r} is not a valid "
            f"regex: {e}"
        ) from e


def _check_scalar(where: str, arg: str, value: object) -> _Scalar:
    if isinstance(value, (str, bool)) or (
        isinstance(value, int) and not isinstance(value, bool)
    ):
        return value
    raise ProgentImportError(
        f"{where}: const/enum value {value!r} for argument {arg!r} is not "
        "a supported scalar (str / int / bool)"
    )


def _translate_restriction(
    where: str, arg: str, restriction: Restriction
) -> tuple[list[_Scalar] | None, str | None]:
    """Translate one restriction to ``(equality alternatives, regex)``.

    Returns ``(values, None)`` for a literal-equality restriction (an
    empty list means it is unsatisfiable), ``(None, pattern)`` for a
    regex restriction, and ``(None, None)`` for a vacuous restriction
    (the empty schema ``{}``).
    """
    if isinstance(restriction, str):
        # Upstream checks bare strings with re.match — anchor at the start.
        _compile(where, arg, restriction)
        return None, r"\A(?:" + restriction + r")"
    unsupported = set(restriction) - _SUPPORTED_KEYWORDS
    if unsupported:
        raise ProgentImportError(
            f"{where}: restriction for argument {arg!r} uses unsupported "
            f"JSON Schema keyword(s) {sorted(unsupported)!r} (supported: "
            f"{sorted(_SUPPORTED_KEYWORDS)!r})"
        )
    type_ = restriction.get("type")
    if "type" in restriction and type_ != "string":
        raise ProgentImportError(
            f"{where}: restriction for argument {arg!r} has unsupported "
            f"type {type_!r} (only 'string' is supported)"
        )
    values: list[_Scalar] | None = None
    if "enum" in restriction:
        enum = restriction["enum"]
        if not isinstance(enum, list) or not enum:
            raise ProgentImportError(
                f"{where}: enum for argument {arg!r} must be a non-empty list"
            )
        values = [_check_scalar(where, arg, v) for v in enum]
    if "const" in restriction:
        const = _check_scalar(where, arg, restriction["const"])
        if values is None:
            values = [const]
        else:
            values = [
                v
                for v in values
                if v == const and isinstance(v, bool) == isinstance(const, bool)
            ]
    pattern = restriction.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ProgentImportError(
                f"{where}: pattern for argument {arg!r} must be a string"
            )
        _compile(where, arg, pattern)
    if type_ == "string" and values is not None:
        values = [v for v in values if isinstance(v, str)]
    if values is not None:
        if pattern is not None:
            # JSON Schema applies string keywords to strings only:
            # non-string enum members pass a pattern check vacuously.
            values = [
                v
                for v in values
                if not isinstance(v, str) or re.search(pattern, v) is not None
            ]
        return values, None
    if pattern is not None:
        return None, pattern
    if type_ == "string":
        # "any string": the empty pattern under re.search semantics.
        return None, ""
    return None, None  # the empty schema {} — vacuous


def _rule_selectors(tool_glob: str, where: str, rule: ProgentRule) -> list[Selector]:
    """Every APG selector alternative for one Progent rule.

    An empty list means the rule's condition is unsatisfiable within the
    subset (e.g. disjoint ``const``/``enum``) and never fires.
    """
    equals_axes: list[tuple[str, list[_Scalar]]] = []
    matches: dict[str, str] = {}
    for arg, restriction in rule.condition.items():
        values, pattern = _translate_restriction(where, arg, restriction)
        if values is not None:
            if not values:
                return []
            equals_axes.append((arg, values))
        elif pattern is not None:
            matches[arg] = pattern
    combos = 1
    for _, values in equals_axes:
        combos *= len(values)
    if combos > MAX_ENUM_EXPANSION:
        raise ProgentImportError(
            f"{where}: enum expansion needs {combos} rule(s), over the "
            f"cap of {MAX_ENUM_EXPANSION}"
        )
    selectors: list[Selector] = []
    for combo in product(*(values for _, values in equals_axes)):
        arg_equals = {
            arg: value for (arg, _), value in zip(equals_axes, combo, strict=True)
        }
        selectors.append(
            Selector(
                tool=tool_glob,
                arg_equals=arg_equals or None,
                arg_matches=dict(matches) or None,
            )
        )
    return selectors


def _fallback_action(fallback: int) -> tuple[Action, str]:
    if fallback == FALLBACK_CONFIRM:
        return Action.REVIEW, "Progent fallback 2: ask the user for confirmation"
    if fallback == FALLBACK_TERMINATE:
        return (
            Action.DENY,
            "Progent fallback 1: terminate (APG denies instead of exiting)",
        )
    return Action.DENY, "Progent fallback 0: refuse with an error message"


def _is_unconditional(selector: Selector) -> bool:
    return not selector.arg_equals and not selector.arg_matches


def convert_progent_policy(
    policy_map: dict[str, list[ProgentRule]],
    *,
    name: str = "progent-import",
    default: str = "deny",
) -> Policy:
    """Translate a parsed Progent policy mapping into an APG :class:`Policy`.

    ``default`` chooses how Progent's "a tool with no entry is denied"
    is carried over: ``"deny"`` (the default) appends a global catch-all
    deny rule, faithful when the converted policy stands alone;
    ``"per-tool"`` omits it so the converted rules can be merged with
    other policies without governing unrelated tools (each governed tool
    still gets its own trailing fall-off-the-end rule either way).
    """
    if default not in ("deny", "per-tool"):
        raise ProgentImportError(
            f"default must be 'deny' or 'per-tool', got {default!r}"
        )
    rules: list[Rule] = []
    for tool, prules in policy_map.items():
        tool_glob = _glob_escape(tool)
        assert fnmatch.fnmatchcase(tool, tool_glob)
        ordered = progent_sorted(prules)
        terminal = False
        last_fallback = FALLBACK_ERROR
        for i, pr in enumerate(ordered):
            if terminal:
                break  # an unconditional rule already decided every call
            where = f"tool {tool!r} rule #{i} (sorted order)"
            last_fallback = pr.fallback
            selectors = _rule_selectors(tool_glob, where, pr)
            if pr.effect == PROGENT_FORBID:
                action, reason = _fallback_action(pr.fallback)
            else:
                action, reason = Action.ALLOW, ""
            for j, sel in enumerate(selectors):
                suffix = f":{j}" if len(selectors) > 1 else ""
                rules.append(
                    Rule(
                        id=f"progent:{tool}:{i}{suffix}",
                        description=(
                            f"Progent rule (priority={pr.priority}, "
                            f"effect={pr.effect}, fallback={pr.fallback})"
                        ),
                        when=sel,
                        effect=Effect(action=action, reason=reason),
                    )
                )
                if _is_unconditional(sel):
                    terminal = True
            if (
                not terminal
                and pr.effect == PROGENT_ALLOW
                and pr.priority == HARD_PRIORITY
                and pr.fallback == FALLBACK_ERROR
            ):
                # Hard allow: a failing argument denies before any later
                # rule is consulted.
                rules.append(
                    Rule(
                        id=f"progent:{tool}:{i}:otherwise",
                        description=(
                            "hard allow (priority 100, fallback 0): any other "
                            f"{tool!r} call is denied immediately"
                        ),
                        when=Selector(tool=tool_glob),
                        effect=Effect(
                            action=Action.DENY,
                            reason="failed a hard (priority 100) Progent allow rule",
                        ),
                    )
                )
                terminal = True
        if not terminal:
            # Fall off the end of the tool's rules. Progent denies via the
            # *last examined rule's* fallback (a leaky loop variable
            # upstream) — an empty rule list denies with an error message.
            action, _ = _fallback_action(last_fallback)
            rules.append(
                Rule(
                    id=f"progent:{tool}:default",
                    description=f"no Progent rule for {tool!r} matched",
                    when=Selector(tool=tool_glob),
                    effect=Effect(
                        action=action,
                        reason=f"the tool '{tool}' is not allowed",
                    ),
                )
            )
    if default == "deny":
        rules.append(
            Rule(
                id="progent:default",
                description="Progent denies any tool without a policy entry",
                when=Selector(),
                effect=Effect(
                    action=Action.DENY,
                    reason="the tool is not allowed (no Progent policy entry)",
                ),
            )
        )
    return Policy(
        name=name,
        description=(
            f"Imported from Progent symbolic rules ({len(policy_map)} tool(s))"
        ),
        rules=tuple(rules),
    )


def _prune(value: object) -> object:
    """Drop ``None``, empty containers and empty description/reason strings."""
    if isinstance(value, dict):
        pruned = {}
        for key, item in value.items():
            item = _prune(item)
            if item is None:
                continue
            if isinstance(item, (dict, list)) and not item:
                continue
            if item == "" and key in ("description", "reason"):
                continue
            pruned[key] = item
        return pruned
    if isinstance(value, list):
        return [_prune(item) for item in value]
    return value


def policy_to_yaml(policy: Policy) -> str:
    """Serialize a :class:`Policy` to YAML loadable by ``load_policy_str``.

    Defaults and empty optional fields are pruned so the output reads
    like a hand-written policy file; the round trip
    ``load_policy_str(policy_to_yaml(p)) == p`` is guaranteed (and
    pinned by the R54 tests).
    """
    data = _prune(policy.model_dump(mode="json"))
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


__all__ = [
    "FALLBACK_CONFIRM",
    "FALLBACK_ERROR",
    "FALLBACK_TERMINATE",
    "HARD_PRIORITY",
    "MAX_ENUM_EXPANSION",
    "PROGENT_ALLOW",
    "PROGENT_FORBID",
    "ProgentImportError",
    "ProgentRule",
    "convert_progent_policy",
    "load_progent_policy",
    "load_progent_policy_str",
    "parse_progent_policy",
    "policy_to_yaml",
    "progent_sorted",
]
