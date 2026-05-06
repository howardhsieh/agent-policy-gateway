"""Tiny in-memory tools used by the indirect-injection example.

These are *deliberately* minimal — the value of the example is the call
graph the gateway sees, not the realism of the implementations. A real
deployment would use a real HTTP client, a real KB, and a real mailer; the
gateway-side wiring would be identical.

Three tools are defined:

* :func:`web_fetch` — returns the body of a URL. The only URL it knows is
  the poisoned page from :mod:`injected_page`; everything else 404s.
* :func:`kb_lookup` — reads from a small in-memory dict that contains one
  intentionally sensitive record so the example has something to exfiltrate.
* :func:`send_email` — appends to an in-memory ``Outbox`` so callers (the
  example runner, tests) can observe what would have been sent without
  needing a real SMTP server.

A factory :func:`build_tools` returns a fresh ``(tools, outbox)`` pair so
each runner gets its own state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from examples.indirect_injection.injected_page import (
    INJECTED_PAGE_HTML,
    INJECTED_PAGE_URL,
)

#: An intentionally sensitive value used to demonstrate exfiltration. In a
#: real system this would be anything from an API token to a customer record.
SECRET_VALUE: str = "TOP-SECRET-LAUNCH-CODES-42"

#: Internal knowledge base records. ``launch-codes`` is the record the
#: injected page tries to extract.
_KB: dict[str, str] = {
    "launch-codes": SECRET_VALUE,
    "office-hours": "Mon-Fri 9-5",
}


@dataclass
class SentEmail:
    """A record of a single ``send_email`` invocation."""

    to: str
    body: str


@dataclass
class Outbox:
    """In-memory mailbox the example writes to in place of a real mailer."""

    sent: list[SentEmail] = field(default_factory=list)

    def append(self, to: str, body: str) -> None:
        self.sent.append(SentEmail(to=to, body=body))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.sent)


@dataclass
class Tools:
    """Bundle of the three tool callables, plus the outbox they write to."""

    web_fetch: Callable[[str], str]
    kb_lookup: Callable[[str], str]
    send_email: Callable[[str, str], dict]
    outbox: Outbox


def build_tools() -> Tools:
    """Construct a fresh tool bundle with its own outbox.

    Each runner (naive / gated) calls this to start from a clean slate so
    their behaviour can be compared side by side.
    """
    outbox = Outbox()

    def web_fetch(url: str) -> str:
        if url == INJECTED_PAGE_URL:
            return INJECTED_PAGE_HTML
        raise ValueError(f"unknown URL in demo: {url!r}")

    def kb_lookup(record: str) -> str:
        if record not in _KB:
            raise KeyError(f"unknown KB record: {record!r}")
        return _KB[record]

    def send_email(to: str, body: str) -> dict:
        outbox.append(to, body)
        return {"to": to, "delivered": True}

    return Tools(
        web_fetch=web_fetch,
        kb_lookup=kb_lookup,
        send_email=send_email,
        outbox=outbox,
    )


__all__ = [
    "Outbox",
    "SECRET_VALUE",
    "SentEmail",
    "Tools",
    "build_tools",
]
