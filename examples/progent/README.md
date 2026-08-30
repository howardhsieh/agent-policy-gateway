# Progent symbolic-rule import demo (R54)

Shows `apg policy import-progent` / `convert_progent_policy` end-to-end:
`rules.json` is a small banking policy written in
[Progent](https://github.com/sunblaze-ucb/progent)'s own runtime format
— `{tool: [[priority, effect, condition, fallback], ...]}` — and the
demo converts it into an ordered first-match APG policy, prints the
generated YAML, and gates six representative calls through a real
`Gateway`.

```bash
python -m examples.progent
# or just the conversion:
apg policy import-progent examples/progent/rules.json
```

Expected outcome — the balance check and the transfer to the allowed UK
IBAN pass; an unlisted amount, a US recipient (the forbid rule), and a
tool with no Progent entry are denied; the recurring schedule (Progent
fallback `2`, "ask the user") is routed to `review`. The entry point
exits non-zero on any other outcome, so it doubles as a CI sanity check.

See `docs/design.md`, "Progent rule import (R54)", for the supported
subset and the documented divergences.
