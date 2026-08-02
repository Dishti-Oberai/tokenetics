"""The one hard-raise type in the project.

Per CLAUDE.md's non-negotiable constraints: the cache-safety guard (stage
7) is the only permitted hard-raise in Tokenetics. Every other failure mode
is fail-open (caught by the orchestrator, logged, request passed through
unmodified). CacheSafetyError is deliberately its own type -- distinct from
every other exception in the codebase -- so a caller can specifically catch
it (`except CacheSafetyError:`) and knows exactly what it means: continuing
would silently invalidate an existing prompt cache.
"""

from __future__ import annotations


class CacheSafetyError(Exception):
    """Raised when the incoming request differs from the caller-supplied
    `previous_request` at or before the last cache breakpoint -- sending it
    as-is would invalidate the cache and silently re-bill everything before
    that point at full price on every subsequent call.
    """
