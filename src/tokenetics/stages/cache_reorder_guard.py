"""Stage 7: cache-aware reorder + cache-safety guard.

Two mechanisms, one pipeline slot (mirrors how stage 4 bundles schema
minification + tool-relevance filtering):

(a) Reorder -- place stable content (system prompt, tool defs) before
volatile content (messages) so the cacheable prefix is maximal. Confirmed
with the user (2026-07-27): today's request shape already emits
system -> tools -> messages in a fixed *JSON key* order by construction
(`to_api_kwargs`). NOTE this is unrelated to (and was once confused with)
Anthropic's actual cache-*hierarchy* order, which is tools -> system ->
messages -- see `cache_breakpoint_optimizer.py`'s 2026-08-02 note for the
anchor-selection bug that confusion caused. With today's shape there's
little to physically reorder yet; this stage still runs as a real
normalize-and-verify step -- defensively re-asserting that invariant --
rather than being skipped, since it's expected to grow real reordering
logic once delta compression or multi-block structures land.

(b) Guard -- THE sole hard-raise in the project (CLAUDE.md non-negotiable).
Diffs the current request's stable prefix (system + tools) against a
caller-supplied `config["previous_request"]` (raw API kwargs, same shape
the caller previously passed to `Tokenetics.prepare()`); if either differs,
raises `CacheSafetyError` instead of silently sending a request that would
invalidate an existing prompt cache and re-bill everything before that
point at full price on every subsequent call. No `previous_request`
supplied means there's nothing to protect yet -- skipped, not guessed at.

The guard only compares system + tools, not the full message history --
this is deliberate and matches the reorder half's scope: those two blocks
are exactly the "stable prefix" this stage protects and the breakpoint
optimizer (stage 8) anchors cache_control to.
"""

from __future__ import annotations

from tokenetics.core.errors import CacheSafetyError
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import TokeneticsRequest, from_api_kwargs


class CacheReorderGuardStage(Stage):
    name = "cache_reorder_guard"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        previous_kwargs = config.get("previous_request")
        if previous_kwargs is not None:
            previous = from_api_kwargs(**previous_kwargs)
            if previous.system != request.system:
                raise CacheSafetyError(
                    "system prompt differs from previous_request -- sending this would "
                    "invalidate the existing cache; sync previous_request or start fresh"
                )
            if previous.tools != request.tools:
                raise CacheSafetyError(
                    "tools differ from previous_request -- sending this would invalidate "
                    "the existing cache; sync previous_request or start fresh"
                )
            self.note(guard_checked=True, guard_result="unchanged")

        # Reorder: system -> tools -> messages is already the fixed order
        # to_api_kwargs emits -- nothing to physically move today, but this
        # is where future reordering logic (once there's more than one
        # stable block to order) plugs in.
        self.note(reorder_noop=True)
        return request
