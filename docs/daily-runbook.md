# Daily run runbook

This document describes what the daily scheduled task does each day. It is also the
contract the task itself follows; if you change this, also update the task prompt.

## On each run

1. **Sync.** `cd` into the repo, `git fetch`, `git pull --rebase` the `main` branch.
2. **Pick.** Open `ROADMAP.md` and pick the **first unchecked item under "Up next"**.
3. **Plan.** Re-read the item's acceptance criteria. Write a short plan (3–6 bullets)
   to `docs/work-log/YYYY-MM-DD.md`.
4. **Implement.** Make the smallest change that satisfies the acceptance criteria.
   Keep diffs reviewable.
5. **Verify.**
   - `python -m pytest -q` must pass.
   - `python -m ruff check .` must pass.
   - If either fails, fix or revert; do not commit broken code.
6. **Document.** Update `CHANGELOG.md` (Keep-a-Changelog format).
7. **Move.** In `ROADMAP.md`, move the completed item from "Up next" to "Done" with the
   date and short commit summary.
8. **Commit & push.** Use a Conventional Commits message: `feat(taint): …`, `docs: …`,
   etc. Push to `main`.
9. **Report.** Write a one-paragraph summary of the day's work to
   `docs/work-log/YYYY-MM-DD.md` and mention the PR/commit URL.

## Failure modes — never push broken code

- If tests/lint fail and cannot be fixed in this run: stop. Leave the repo clean
  (`git stash` or revert), write the work-log noting the blocker, and **do not push**.
- If GitHub auth fails: stop. Write the work-log noting the auth issue.
- If the chosen roadmap item turns out to be too big for one run: split it into smaller
  items in `ROADMAP.md` (commit just the roadmap change), pick the first sub-item, and
  proceed.

## Categorizing outputs

All design notes, work logs, threat models, and benchmark write-ups live under `docs/`,
organized by purpose:

- `docs/design.md` — architecture
- `docs/threat-model.md` — adversary classes and assumptions (created by R13)
- `docs/work-log/YYYY-MM-DD.md` — one file per daily run
- `docs/benchmarks/` — performance notes (created by R12)
- `docs/adapters/` — per-protocol notes for MCP/OpenAI/Anthropic adapters

Code lives under `src/`, tests under `tests/`, runnable demos under `examples/`.
