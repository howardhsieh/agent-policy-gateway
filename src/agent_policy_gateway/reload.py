"""Policy hot-reload / file-watch (R20).

A :class:`~agent_policy_gateway.policy.Policy` is immutable and loaded once
from disk by :func:`~agent_policy_gateway.policy.load_policy`. That is the
right default for a short-lived process, but a long-lived
:class:`~agent_policy_gateway.gateway.Gateway` — a sidecar, a server, a
notebook kernel — wants to pick up edits to its policy file *without*
restarting.

:class:`WatchedPolicy` provides that, opt-in, with a deliberately small
surface. It wraps a single policy file and is **duck-typed** to expose the
one method the reference monitor uses — ``first_match(call, *, resource=...)``
— so it can be dropped straight into ``Gateway.policies`` with no change to
the gateway itself::

    from agent_policy_gateway import Gateway, watch_policy

    gw = Gateway(policies=[watch_policy("policies/default.yaml")])
    # ... edit policies/default.yaml on disk ...
    # the next call through `gw` sees the new rules automatically.

Reload semantics:

* **Cheap check, lazy reload.** Every ``first_match`` stats the file and
  compares a ``(st_mtime_ns, st_size)`` signature against the last one it
  loaded. Unchanged file → no work. Changed file → re-read and validate.
* **Fail-closed on a bad edit.** If the new file is invalid YAML or fails
  schema validation, the previously-loaded policy keeps serving. The parse
  error is logged through ``logging.getLogger("agent_policy_gateway.reload")``
  and handed to an optional ``on_error`` callback. The bad signature is
  remembered so a broken file is not re-parsed on every single call — it is
  retried only once the file changes again (e.g. the operator fixes it).
* **Errors at construction still raise.** ``watch_policy`` performs the
  initial load eagerly, so a typo in the *starting* file surfaces
  immediately rather than silently serving an empty policy.

This module performs no background threads and installs no OS file-watch
hooks; the check is driven synchronously by calls passing through the
gateway. That keeps it dependency-free and easy to reason about — there is
never a window where a half-written file is observed mid-call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from agent_policy_gateway.core import ToolCall
from agent_policy_gateway.policy import Policy, PolicyError, Rule, load_policy

logger = logging.getLogger("agent_policy_gateway.reload")

# A file signature: (modification time in ns, size in bytes). Comparing this
# tuple catches both ordinary edits and same-timestamp edits that change the
# length, without hashing the contents on every call.
_Signature = tuple[int, int]

OnError = Callable[[Exception], None]


def _signature(path: Path) -> _Signature:
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


class WatchedPolicy:
    """A :class:`Policy` view that reloads itself when its file changes.

    Duck-typed to :meth:`Policy.first_match` so it can live in
    ``Gateway.policies`` alongside plain policies. Construct via
    :func:`watch_policy` (which loads eagerly) or directly.

    Args:
        path: Path to the policy YAML file.
        on_error: Optional callback invoked with the exception when a reload
            fails. The previous policy keeps serving regardless; this is for
            surfacing the failure (metrics, alerts) on top of the log line.

    Attributes are intentionally private — the only supported reads are
    :meth:`first_match` (the gateway contract) and the :attr:`policy`
    property (handy for tests and introspection).
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | Path,
        *,
        on_error: OnError | None = None,
    ) -> None:
        self._path = Path(path)
        self._on_error = on_error
        # Eager initial load: a broken starting file should fail loudly here
        # rather than serving an empty policy.
        self._policy: Policy = load_policy(self._path)
        self._loaded_sig: _Signature = _signature(self._path)
        # The signature of a file we already tried and rejected, so we do not
        # re-parse a known-bad file on every call. Reset whenever the file
        # changes again.
        self._failed_sig: _Signature | None = None

    # ----- reload machinery -----------------------------------------------------

    def maybe_reload(self) -> bool:
        """Reload the policy if the file changed since the last load.

        Returns ``True`` when a new policy was successfully swapped in,
        ``False`` otherwise (file unchanged, missing, or a rejected edit).
        Never raises: a failed reload is logged and the previous policy
        stays active (fail-closed).
        """
        try:
            sig = _signature(self._path)
        except OSError as e:
            # File vanished or became unreadable mid-run. Keep serving the
            # last good policy; log once per distinct failure.
            self._handle_error(e)
            return False

        if sig == self._loaded_sig:
            return False
        if sig == self._failed_sig:
            # Already tried this exact bad version; don't spam parse attempts.
            return False

        try:
            new_policy = load_policy(self._path)
        except (PolicyError, OSError) as e:
            self._failed_sig = sig
            self._handle_error(e)
            return False

        self._policy = new_policy
        self._loaded_sig = sig
        self._failed_sig = None
        logger.info("reloaded policy from %s", self._path)
        return True

    def _handle_error(self, exc: Exception) -> None:
        logger.warning(
            "policy reload failed for %s; keeping previous policy: %s",
            self._path,
            exc,
        )
        if self._on_error is not None:
            self._on_error(exc)

    # ----- Policy duck-typing ----------------------------------------------------

    @property
    def policy(self) -> Policy:
        """The currently-active :class:`Policy`, reloading first if needed."""
        self.maybe_reload()
        return self._policy

    @property
    def path(self) -> Path:
        """The watched policy file path."""
        return self._path

    def first_match(
        self,
        call: ToolCall,
        *,
        resource: str | None = None,
    ) -> Rule | None:
        """Reload if the file changed, then delegate to the live policy.

        Matches :meth:`Policy.first_match` exactly so a ``WatchedPolicy`` is
        a drop-in member of ``Gateway.policies``.
        """
        self.maybe_reload()
        return self._policy.first_match(call, resource=resource)


def watch_policy(
    path: str | os.PathLike[str] | Path,
    *,
    on_error: OnError | None = None,
) -> WatchedPolicy:
    """Build a :class:`WatchedPolicy` for ``path`` (loads eagerly).

    Convenience mirror of :func:`agent_policy_gateway.policy.load_policy` for
    the long-lived case: the returned object reloads itself whenever the file
    changes, falling back to the last good policy on an invalid edit.
    """
    return WatchedPolicy(path, on_error=on_error)


__all__ = [
    "OnError",
    "WatchedPolicy",
    "watch_policy",
]
