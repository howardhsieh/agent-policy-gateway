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

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``apg`` console script.

    Returns the subcommand's exit code: ``0`` success, ``1`` invalid policy,
    ``2`` missing file.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
