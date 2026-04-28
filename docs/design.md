# Design

## Position in the agent stack

```
+----------------+         +-------------------+         +-----------+
|   LLM / Agent  |  --->   |  Policy Gateway   |  --->   |   Tool    |
+----------------+         +-------------------+         +-----------+
                            ^         ^      ^
                            |         |      |
                       policies   audit log  taint store
```

The gateway is a **reference monitor**: every tool call passes through it, every decision
is logged, and policies cannot be bypassed by the LLM.

## Information flow control (IFC)

We borrow the classic lattice model from OS-level IFC: each piece of data carries a label,
labels form a lattice under join (`∨`) and order (`⊑`), and the gateway enforces
non-interference rules at sinks.

Concretely, every tool output is tagged with a set of *source* labels — strings like
`web`, `user_upload`, `crm.contact.email`. When tool A's output is passed as an argument
to tool B, B's effective input label is the join of all argument labels. Policies on B can
then refuse, require human review, or downgrade based on the input label.

This is a coarse approximation of full IFC — we don't track field-level taint inside
JSON outputs (yet) — but it's enough to catch the dominant exfiltration patterns:

- *Indirect prompt injection*: a malicious web page tells the agent to email the user's
  contacts. Without IFC, the email send looks fine. With IFC, the send's `to` and `body`
  carry `web` taint, and the policy refuses.

## Why a gateway, not an LLM-side guard

The model is adversarially-influenced by tool outputs. Anything we ask the model to do as
self-defense is bypassable. The gateway is outside the model's control surface, so the
guarantee is structural rather than emergent.


## Declassification

Some tools are trusted to *strip* a source label from their output. A vetted PII
redactor that scrubs identifiers should be allowed to remove the `pii` label;
without an explicit declassification mechanism, the only escape from a once-tainted
flow is to refuse it forever, which makes the gateway useless in practice.

We model this as a `ToolTaintSpec(adds, declassifies)`: every call's output label
is `((∨ inputs) ∨ adds) \ declassifies`. Declassification is a privilege a tool
declares once and the operator audits; it is intentionally *not* something the
LLM can request at runtime.

## Open questions

- Field-level taint inside structured tool outputs.
- Policy-as-code (Python) vs. data (YAML/JSON). Currently leaning data with a small set
  of well-defined operators.
- Streaming tool outputs and incremental taint propagation.
