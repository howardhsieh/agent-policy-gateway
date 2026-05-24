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

``apg policy explain <file> --tool NAME [--identity ID] [--taint a,b] [--resource R]``
    Build a hypothetical :class:`~agent_policy_gateway.core.ToolCall` from the
    given selectors and walk the policy's rules in declaration order, printing a
    *first-match trace*: for each rule, whether it matched and (when it did not)
    which selector clause rejected it. Stops at the first matching rule and
    prints its id and effect. When no rule matches, prints the line noting that
    the gateway's default disposition applies.

The module is otherwise pure: the only I/O is reading the policy file (delegated
to :func:`load_policy`) and writing to stdout/stderr.
"""

from __future__ import annotations

import argparse
import fnmatch
import sys

from agent_policy_gateway.core import TaintLabel, ToolCall
from agent_policy_gateway.policy import Policy, PolicyError, Selector, load_policy


def _parse_taint(raw: str | None) -> TaintLabel:
    """Build a :class:`TaintLabel` from a comma-separated ``--taint`` value."""
    if not raw:
        return TaintLabel()
    sources = [s.strip() for s in raw.split(",") if s.strip()]
    return TaintLabel.of(*sources)


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
    lines.append(
        f"call:   tool={call.tool_name!r} identity={call.agent_id!r} "
        f"taint={taint!r} resource={resource!r}"
    )
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
    )
    for line in _explain(policy, call, resource=args.resource):
        print(line)
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
    explain_p.set_defaults(func=_cmd_explain)

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
