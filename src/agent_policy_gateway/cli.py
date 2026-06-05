"""``apg`` console entry point: policy ``validate`` and ``explain`` (R18).

This module wires a small, dependency-free :mod:`argparse` CLI on top of the
existing policy machinery in :mod:`agent_policy_gateway.policy`. It mirrors the
style of the other entry points (``apg-replay`` in :mod:`audit`, ``apg-bench``
in :mod:`bench`): a ``main(argv)`` that returns an ``int`` exit code so it is
trivially unit-testable, with the real script shim under ``if __name__``.

Subcommands
-----------
``apg policy validate <file>``
    Parse and Pydantic-validate a policy YAML. Prints a one-line OK summary and
    exits ``0`` when the policy is well-formed; exits ``2`` when the file does
    not exist; exits ``1`` with the friendly, line-located error message that
    :func:`~agent_policy_gateway.policy.load_policy_str` already constructs from
    ``yaml.YAMLError.problem_mark`` / Pydantic ``ValidationError``.

``apg policy explain <file> --tool NAME [--identity ID] [--taint a,b] [--resource R] [--arg K=V]``
    Build a hypothetical :class:`~agent_policy_gateway.core.ToolCall` from the
    given selectors and walk the policy's rules in declaration order, printing a
    *first-match trace*: for each rule, whether it matched and (when it did not)
    which selector clause rejected it. Stops at the first matching rule and
    prints its id and effect. When no rule matches, prints the line noting that
    the gateway's default disposition applies.

``apg policy diff <old> <new> [--tool NAME] [--identity ID] [--taint a,b] [--resource R]``
    Compare two policy files by *decisions* rather than text (R24). A matrix of
    synthetic tool calls is derived from both policies' rule selectors (each
    selector is concretized into one scenario that satisfies it, plus a neutral
    baseline), every scenario is evaluated against both policies with
    :meth:`~agent_policy_gateway.policy.Policy.first_match`, and scenarios whose
    ``(rule id, action)`` outcome changed are reported. Supplying any of
    ``--tool/--identity/--taint/--resource`` replaces the matrix with that
    single user-defined scenario, mirroring ``explain``. Exits ``0`` whether or
    not changes were found (printing ``no decision changes`` when none were),
    ``2`` when either file is missing, ``1`` when either policy is malformed.

``apg policy lint <file>``
    Static quality checks on a single policy file (R26). Reports rules that can
    *never* match (a self-contradictory taint clause, e.g. a source required by
    ``all_of`` but forbidden by ``none_of``) and rules that are *shadowed* by an
    earlier rule whose selector is at least as general (first-match means the
    later rule is dead). Shadowing detection is deliberately conservative:
    findings are emitted only when generality is certain, so a clean policy may
    still contain subtle dead rules, but every reported finding is real. Exits
    ``0`` when no findings, ``3`` when findings were reported, ``2`` when the
    file is missing, ``1`` when the policy is malformed.

The module is otherwise pure: the only I/O is reading the policy file (delegated
to :func:`load_policy`) and writing to stdout/stderr.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys

from agent_policy_gateway.core import TaintLabel, ToolCall
from agent_policy_gateway.policy import (
    Policy,
    PolicyError,
    Selector,
    TaintCondition,
    _arg_value_equal,
    load_policy,
)


def _parse_taint(raw: str | None) -> TaintLabel:
    """Build a :class:`TaintLabel` from a comma-separated ``--taint`` value."""
    if not raw:
        return TaintLabel()
    sources = [s.strip() for s in raw.split(",") if s.strip()]
    return TaintLabel.of(*sources)


def _coerce_scalar(value: str) -> str | int | bool:
    """Interpret a ``--arg`` value the way YAML would a plain scalar.

    ``true`` / ``false`` (case-insensitive) become bools, decimal integers
    become ints, everything else stays a string. Deliberately *not* YAML
    parsing: ``yaml.safe_load("#public")`` would read a comment and return
    ``None``, which is exactly the kind of value a channel argument needs.
    """
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value


def _arg_pair(raw: str) -> tuple[str, str | int | bool]:
    """argparse type for ``--arg KEY=VALUE`` (repeatable on explain)."""
    key, sep, value = raw.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    return key, _coerce_scalar(value)


def _clause_rejection(
    selector: Selector,
    call: ToolCall,
    *,
    resource: str | None,
) -> str | None:
    """Return a human reason the selector rejected the call, or ``None``.

    ``None`` means every clause was satisfied (the selector matches). The
    checks mirror :meth:`Selector.matches` exactly so the trace never disagrees
    with the real first-match used by the gateway.
    """
    if selector.tool is not None and not fnmatch.fnmatchcase(call.tool_name, selector.tool):
        return f"tool {call.tool_name!r} does not match glob {selector.tool!r}"
    if selector.identity is not None and call.agent_id != selector.identity:
        return f"identity {call.agent_id!r} != required {selector.identity!r}"
    if selector.resource is not None:
        if resource is None:
            return f"rule needs resource matching {selector.resource!r} but none was given"
        if not fnmatch.fnmatchcase(resource, selector.resource):
            return f"resource {resource!r} does not match glob {selector.resource!r}"
    if selector.arg_equals:
        for key, expected in selector.arg_equals.items():
            if key not in call.args:
                return f"argument {key!r} is missing (rule needs {key}={expected!r})"
            if not _arg_value_equal(expected, call.args[key]):
                return f"argument {key}={call.args[key]!r} != required {expected!r}"
    if selector.taint is not None and not selector.taint.matches(call.input_label):
        return (
            f"taint {sorted(call.input_label.sources)!r} fails condition "
            f"(any_of={list(selector.taint.any_of)}, all_of={list(selector.taint.all_of)}, "
            f"none_of={list(selector.taint.none_of)})"
        )
    return None


def _explain(policy: Policy, call: ToolCall, *, resource: str | None) -> list[str]:
    """Render the first-match trace for ``call`` against ``policy`` as lines."""
    lines: list[str] = []
    taint = sorted(call.input_label.sources)
    lines.append(
        f"policy: {policy.name!r} ({len(policy.rules)} rule(s))"
    )
    call_line = (
        f"call:   tool={call.tool_name!r} identity={call.agent_id!r} "
        f"taint={taint!r} resource={resource!r}"
    )
    if call.args:
        call_line += f" args={call.args!r}"
    lines.append(call_line)
    lines.append("first-match trace:")
    matched_id: str | None = None
    for rule in policy.rules:
        if matched_id is not None:
            lines.append(f"  [ skip ] {rule.id}: (already matched earlier)")
            continue
        reason = _clause_rejection(rule.when, call, resource=resource)
        if reason is None:
            matched_id = rule.id
            lines.append(f"  [MATCH ] {rule.id}: selector satisfied")
        else:
            lines.append(f"  [ no  ] {rule.id}: {reason}")
    lines.append("")
    if matched_id is not None:
        rule = next(r for r in policy.rules if r.id == matched_id)
        eff = rule.effect
        detail = f"action={eff.action.value}"
        if eff.reason:
            detail += f" reason={eff.reason!r}"
        lines.append(f"=> matched rule: {matched_id} ({detail})")
    else:
        lines.append(
            "=> no rule matched; the gateway's default disposition applies "
            "to this call"
        )
    return lines


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        policy = load_policy(args.file)
    except FileNotFoundError:
        print(f"apg: policy file not found: {args.file}", file=sys.stderr)
        return 2
    except PolicyError as exc:
        print(f"apg: invalid policy: {exc}", file=sys.stderr)
        return 1
    print(f"OK: policy {policy.name!r} is valid ({len(policy.rules)} rule(s))")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    try:
        policy = load_policy(args.file)
    except FileNotFoundError:
        print(f"apg: policy file not found: {args.file}", file=sys.stderr)
        return 2
    except PolicyError as exc:
        print(f"apg: invalid policy: {exc}", file=sys.stderr)
        return 1
    call = ToolCall(
        tool_name=args.tool,
        agent_id=args.identity,
        input_label=_parse_taint(args.taint),
        args=dict(args.arg or []),
    )
    for line in _explain(policy, call, resource=args.resource):
        print(line)
    return 0



# --- ``apg policy diff`` (R24) -------------------------------------------------

#: A synthetic scenario: (tool_name, identity, taint sources, resource).
_Scenario = tuple[str, str | None, frozenset[str], str | None]

#: Neutral probe values for unconstrained selector fields. Deliberately odd
#: names so they do not accidentally satisfy unrelated rules' globs.
_ANY_TOOL = "__any_tool__"


def _concretize_glob(glob: str | None, *, default: str | None) -> str | None:
    """Return a concrete sample value satisfying the fnmatch ``glob``.

    ``None`` (an unconstrained selector field) yields ``default``. Wildcard
    characters are substituted with a literal ``x``; if the substitution does
    not actually satisfy the glob (exotic patterns), the glob text itself is
    returned as a best effort.
    """
    if glob is None:
        return default
    candidate = glob.replace("*", "x").replace("?", "x")
    if fnmatch.fnmatchcase(candidate, glob):
        return candidate
    return glob


def _scenarios_from_policy(policy: Policy) -> list[_Scenario]:
    """One synthetic scenario per rule, concretized from its selector."""
    scenarios: list[_Scenario] = []
    for rule in policy.rules:
        sel = rule.when
        tool = _concretize_glob(sel.tool, default=_ANY_TOOL)
        assert tool is not None  # default is non-None for tools
        sources: set[str] = set()
        if sel.taint is not None:
            sources.update(sel.taint.all_of)
            if sel.taint.any_of and not (set(sel.taint.any_of) & sources):
                pick = next(
                    (s for s in sel.taint.any_of if s not in sel.taint.none_of),
                    None,
                )
                if pick is not None:
                    sources.add(pick)
        resource = _concretize_glob(sel.resource, default=None)
        scenarios.append((tool, sel.identity, frozenset(sources), resource))
    return scenarios


def _build_matrix(old: Policy, new: Policy) -> list[_Scenario]:
    """Deduplicated scenario matrix: neutral baseline + both policies' rules."""
    baseline: _Scenario = (_ANY_TOOL, None, frozenset(), None)
    matrix: list[_Scenario] = []
    seen: set[_Scenario] = set()
    for sc in [baseline, *_scenarios_from_policy(old), *_scenarios_from_policy(new)]:
        if sc not in seen:
            seen.add(sc)
            matrix.append(sc)
    return matrix


def _decide(policy: Policy, scenario: _Scenario) -> tuple[str, str] | None:
    """First-match decision for ``scenario``: ``(rule_id, action)`` or ``None``."""
    tool, identity, sources, resource = scenario
    label = TaintLabel.of(*sorted(sources)) if sources else TaintLabel()
    call = ToolCall(tool_name=tool, agent_id=identity, input_label=label)
    rule = policy.first_match(call, resource=resource)
    if rule is None:
        return None
    return (rule.id, rule.effect.action.value)


def _format_decision(decision: tuple[str, str] | None) -> str:
    if decision is None:
        return "(no match - default disposition)"
    rule_id, action = decision
    return f"{rule_id} -> {action}"


def _cmd_diff(args: argparse.Namespace) -> int:
    policies: list[Policy] = []
    for label, path in (("old", args.old), ("new", args.new)):
        try:
            policies.append(load_policy(path))
        except FileNotFoundError:
            print(f"apg: {label} policy file not found: {path}", file=sys.stderr)
            return 2
        except PolicyError as exc:
            print(f"apg: invalid {label} policy: {exc}", file=sys.stderr)
            return 1
    old, new = policies

    single = any(
        v is not None for v in (args.tool, args.identity, args.taint, args.resource)
    )
    if single:
        matrix: list[_Scenario] = [
            (
                args.tool if args.tool is not None else _ANY_TOOL,
                args.identity,
                frozenset(_parse_taint(args.taint).sources),
                args.resource,
            )
        ]
    else:
        matrix = _build_matrix(old, new)

    changes: list[tuple[_Scenario, tuple[str, str] | None, tuple[str, str] | None]] = []
    for sc in matrix:
        old_d = _decide(old, sc)
        new_d = _decide(new, sc)
        if old_d != new_d:
            changes.append((sc, old_d, new_d))

    print(
        f"comparing policies: {old.name!r} ({len(old.rules)} rule(s)) -> "
        f"{new.name!r} ({len(new.rules)} rule(s))"
    )
    if not changes:
        print(f"no decision changes across {len(matrix)} scenario(s)")
        return 0
    print(f"{len(changes)} decision change(s) across {len(matrix)} scenario(s):")
    for (tool, identity, sources, resource), old_d, new_d in changes:
        print(
            f"  - tool={tool!r} identity={identity!r} "
            f"taint={sorted(sources)!r} resource={resource!r}"
        )
        print(f"      old: {_format_decision(old_d)}")
        print(f"      new: {_format_decision(new_d)}")
    return 0


# --- ``apg policy lint`` (R26) -------------------------------------------------

#: Characters that make an fnmatch pattern non-literal.
_GLOB_CHARS = frozenset("*?[")


def _pattern_at_least_as_general(earlier: str | None, later: str | None) -> bool:
    """Conservatively decide whether glob field ``earlier`` subsumes ``later``.

    Returns True only when *every* value matched by ``later`` is certainly
    matched by ``earlier``: an unset earlier field constrains nothing; equal
    patterns subsume each other; and a *literal* later value (no glob
    metacharacters) can be tested directly with :func:`fnmatch.fnmatchcase`.
    Two distinct globs are never compared (glob-subsumption in general is not
    worth the subtlety here) — the check stays conservative and never
    produces a false positive.
    """
    if earlier is None:
        return True
    if later is None:
        return False
    if earlier == later:
        return True
    if not (_GLOB_CHARS & set(later)):
        return fnmatch.fnmatchcase(later, earlier)
    return False


def _args_at_least_as_general(
    earlier: dict[str, str | int | bool] | None,
    later: dict[str, str | int | bool] | None,
) -> bool:
    """True iff ``earlier``'s ``arg_equals`` constraints subsume ``later``'s.

    Fewer constraints are more general: every (key, value) pair the earlier
    selector requires must also be required — with an equal value, type-strict
    via :func:`_arg_value_equal` — by the later selector.
    """
    if not earlier:
        return True
    if not later:
        return False
    return all(
        key in later and _arg_value_equal(value, later[key])
        for key, value in earlier.items()
    )


def _taint_at_least_as_general(
    earlier: TaintCondition | None, later: TaintCondition | None
) -> bool:
    """True iff every label satisfying ``later`` certainly satisfies ``earlier``.

    Conservative sufficient conditions, mirroring ``TaintCondition.matches``:
    ``earlier.all_of`` must be a subset of ``later.all_of`` (the later rule
    already guarantees those sources are present); ``earlier.none_of`` must be
    a subset of ``later.none_of`` (the later rule already guarantees their
    absence); and a non-empty ``earlier.any_of`` must be guaranteed either by
    a source the later rule requires via ``all_of`` or because every source
    ``later.any_of`` can supply is accepted by ``earlier.any_of``.
    """
    if earlier is None or earlier.is_empty():
        return True
    if later is None or later.is_empty():
        return False
    if not set(earlier.all_of) <= set(later.all_of):
        return False
    if not set(earlier.none_of) <= set(later.none_of):
        return False
    if earlier.any_of:
        guaranteed_by_all = bool(set(earlier.any_of) & set(later.all_of))
        guaranteed_by_any = bool(later.any_of) and set(later.any_of) <= set(
            earlier.any_of
        )
        if not (guaranteed_by_all or guaranteed_by_any):
            return False
    return True


def _selector_at_least_as_general(earlier: Selector, later: Selector) -> bool:
    """True iff ``earlier`` certainly matches every call ``later`` matches."""
    return (
        _pattern_at_least_as_general(earlier.tool, later.tool)
        and (earlier.identity is None or earlier.identity == later.identity)
        and _pattern_at_least_as_general(earlier.resource, later.resource)
        and _args_at_least_as_general(earlier.arg_equals, later.arg_equals)
        and _taint_at_least_as_general(earlier.taint, later.taint)
    )


def _taint_contradiction(cond: TaintCondition | None) -> str | None:
    """Explain why ``cond`` is unsatisfiable, or None when it is satisfiable."""
    if cond is None:
        return None
    clash = set(cond.all_of) & set(cond.none_of)
    if clash:
        source = sorted(clash)[0]
        return (
            f"taint clause requires source {source!r} in all_of "
            "but also forbids it in none_of"
        )
    if cond.any_of and set(cond.any_of) <= set(cond.none_of):
        return (
            f"taint clause requires one of any_of {sorted(cond.any_of)!r} "
            "but none_of forbids every one of them"
        )
    return None


def _lint(policy: Policy) -> list[str]:
    """Run the static checks and return the findings, in rule order."""
    findings: list[str] = []
    contradictory: set[str] = set()
    for index, rule in enumerate(policy.rules):
        why = _taint_contradiction(rule.when.taint)
        if why is not None:
            contradictory.add(rule.id)
            findings.append(f"W002 rule {rule.id!r} can never match: {why}")
            continue
        for earlier in policy.rules[:index]:
            if earlier.id in contradictory:
                continue
            if _selector_at_least_as_general(earlier.when, rule.when):
                findings.append(
                    f"W001 rule {rule.id!r} is shadowed by earlier rule "
                    f"{earlier.id!r}: every call it matches is matched "
                    f"first by {earlier.id!r}"
                )
                break
    return findings


def _cmd_lint(args: argparse.Namespace) -> int:
    try:
        policy = load_policy(args.file)
    except FileNotFoundError:
        print(f"apg: policy file not found: {args.file}", file=sys.stderr)
        return 2
    except PolicyError as exc:
        print(f"apg: invalid policy: {exc}", file=sys.stderr)
        return 1
    findings = _lint(policy)
    if not findings:
        print(
            f"OK: no lint findings in policy {policy.name!r} "
            f"({len(policy.rules)} rule(s))"
        )
        return 0
    for line in findings:
        print(line)
    print(f"{len(findings)} lint finding(s) in policy {policy.name!r}")
    return 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apg",
        description="agent-policy-gateway command-line tools.",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    policy_p = sub.add_parser(
        "policy",
        help="Inspect and validate policy files.",
        description="Validate a policy file or explain which rule a call hits.",
    )
    policy_sub = policy_p.add_subparsers(dest="command", required=True)

    validate_p = policy_sub.add_parser(
        "validate",
        help="Validate a policy YAML against the schema.",
        description=(
            "Parse and validate a policy YAML. Exits 0 if valid, 2 if the file "
            "is missing, 1 with a line-located message if it is malformed."
        ),
    )
    validate_p.add_argument("file", help="Path to the policy YAML file.")
    validate_p.set_defaults(func=_cmd_validate)

    explain_p = policy_sub.add_parser(
        "explain",
        help="Show which rule a hypothetical call would match, and why.",
        description=(
            "Build a tool call from --tool/--identity/--taint/--resource and "
            "print the first-match trace through the policy's rules."
        ),
    )
    explain_p.add_argument("file", help="Path to the policy YAML file.")
    explain_p.add_argument(
        "--tool",
        required=True,
        metavar="NAME",
        help="Tool name of the hypothetical call (e.g. send_email).",
    )
    explain_p.add_argument(
        "--identity",
        default=None,
        metavar="ID",
        help="Agent identity making the call (matched against rule identity).",
    )
    explain_p.add_argument(
        "--taint",
        default=None,
        metavar="a,b,c",
        help="Comma-separated taint sources on the call input (e.g. web,pii).",
    )
    explain_p.add_argument(
        "--resource",
        default=None,
        metavar="R",
        help="Target resource of the call (matched against rule resource globs).",
    )
    explain_p.add_argument(
        "--arg",
        action="append",
        default=None,
        type=_arg_pair,
        metavar="KEY=VALUE",
        help=(
            "Argument on the hypothetical call, repeatable "
            "(e.g. --arg channel=#public --arg dry_run=true). Values true/false "
            "become bools, decimal integers become ints, all else stays a string."
        ),
    )
    explain_p.set_defaults(func=_cmd_explain)

    diff_p = policy_sub.add_parser(
        "diff",
        help="Compare two policy files by the decisions they produce.",
        description=(
            "Evaluate a matrix of synthetic tool calls (derived from both "
            "policies' rule selectors, or a single --tool/--identity/--taint/"
            "--resource scenario) against both policies and report scenarios "
            "whose first-match decision changed. Exits 0 either way, 2 if a "
            "file is missing, 1 if a policy is malformed."
        ),
    )
    diff_p.add_argument("old", help="Path to the old policy YAML file.")
    diff_p.add_argument("new", help="Path to the new policy YAML file.")
    diff_p.add_argument(
        "--tool",
        default=None,
        metavar="NAME",
        help="Restrict the diff to a single scenario with this tool name.",
    )
    diff_p.add_argument(
        "--identity",
        default=None,
        metavar="ID",
        help="Agent identity for the single-scenario diff.",
    )
    diff_p.add_argument(
        "--taint",
        default=None,
        metavar="a,b,c",
        help="Comma-separated taint sources for the single-scenario diff.",
    )
    diff_p.add_argument(
        "--resource",
        default=None,
        metavar="R",
        help="Target resource for the single-scenario diff.",
    )
    diff_p.set_defaults(func=_cmd_diff)

    lint_p = policy_sub.add_parser(
        "lint",
        help="Report dead rules: shadowed or self-contradictory selectors.",
        description=(
            "Run static quality checks on a policy file. Reports rules that "
            "can never match (self-contradictory taint clause) and rules "
            "shadowed by an earlier, at-least-as-general rule. Exits 0 when "
            "clean, 3 when findings were reported, 2 if the file is missing, "
            "1 if the policy is malformed."
        ),
    )
    lint_p.add_argument("file", help="Path to the policy YAML file.")
    lint_p.set_defaults(func=_cmd_lint)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``apg`` console script.

    Returns the subcommand's exit code: ``0`` success, ``1`` invalid policy,
    ``2`` missing file, ``3`` lint findings (``policy lint`` only).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
