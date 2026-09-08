import asyncio
import time
from dataclasses import dataclass, field

from onecard.errors import BudgetError, EvictionError
from onecard.gpu.consumer import GpuConsumer, Residency


@dataclass
class SwapEvent:
    evicted: str
    reason: str
    duration_s: float


@dataclass
class _Held:
    residency: Residency
    last_used: float = field(default_factory=time.monotonic)


class Arbiter:
    """Owns the GPU. Decides who holds it and enforces the VRAM budget.

    Every consumer of the card goes through here. The arbiter reasons about
    claims, never about model families.
    """

    def __init__(self, budget_mb: int, consumers: dict[str, GpuConsumer]) -> None:
        self.budget_mb = budget_mb
        self.consumers = consumers
        self.swaps: list[SwapEvent] = []
        self._held: dict[tuple[str, str], _Held] = {}
        self._lock = asyncio.Lock()
        # Adopted models get strictly decreasing (and always negative) last_used
        # stamps, so they sort older than anything time.monotonic() can produce
        # and older than each other in discovery order.
        self._adopt_clock = 0.0

    async def residents(self) -> list[Residency]:
        return [h.residency for h in self._held.values()]

    async def used_mb(self) -> int:
        return sum(h.residency.footprint_mb for h in self._held.values())

    async def claim(
        self,
        consumer: str,
        key: str,
        need_mb: int,
        *,
        pinned: bool = False,
        exclusive: bool = False,
    ) -> Residency:
        if consumer not in self.consumers:
            raise KeyError(f"unknown GPU consumer: {consumer}")
        async with self._lock:
            return await self._claim_locked(
                consumer, key, need_mb, pinned=pinned, exclusive=exclusive
            )

    async def _claim_locked(
        self, consumer: str, key: str, need_mb: int, *, pinned: bool, exclusive: bool
    ) -> Residency:
        if need_mb > self.budget_mb:
            raise BudgetError(
                f"'{key}' needs {need_mb}MB which exceeds the budget of {self.budget_mb}MB"
            )

        ident = (consumer, self.consumers[consumer].key_for(key))
        reusable = self._reusable(ident, pinned=pinned)
        if not exclusive and reusable is not None:
            reusable.last_used = time.monotonic()
            return reusable.residency

        if exclusive:
            await self._release_everything(reason=f"exclusive claim by '{key}'")
        else:
            # Nothing releases the card at process exit: Ollama holds a model
            # for its keep_alive after we are gone. A fresh Arbiter that
            # trusted its own empty bookkeeping would load on top of that and
            # spill to system RAM, which is the one failure this project
            # exists to prevent. So find out what the card really holds before
            # doing any budget arithmetic.
            await self._reconcile_locked()
            reusable = self._reusable(ident, pinned=pinned)
            if reusable is not None:
                reusable.last_used = time.monotonic()
                return reusable.residency

        while await self.used_mb() + need_mb > self.budget_mb:
            victim = self._lru_victim()
            if victim is None:
                raise BudgetError(
                    f"'{key}' needs {need_mb}MB and cannot fit: "
                    f"{await self.used_mb()}MB is held by models that cannot be evicted"
                )
            await self._evict(victim, reason=f"making room for '{key}'")

        residency = await self.consumers[consumer].load(
            key, pinned=pinned, footprint_hint_mb=need_mb
        )
        self._held[ident] = _Held(residency=residency)
        return residency

    def _reusable(self, ident: tuple[str, str], *, pinned: bool) -> _Held | None:
        """The held entry that already satisfies this claim, if there is one.

        A claim for a pinned model is not satisfied by an unpinned entry: it
        has to be loaded again with a pinning keep_alive, or it would stay
        evictable while validation reserves its footprint forever.
        """
        held = self._held.get(ident)
        if held is None:
            return None
        if pinned and not held.residency.pinned:
            return None
        return held

    async def _reconcile_locked(self) -> None:
        """Adopt models the card holds that this process did not load.

        Adopted entries are unpinned, so they are evictable, and carry the
        footprint the consumer reports -- including models the config never
        declared, because the budget is about the card, not about our config.
        A BackendError from a consumer propagates: proceeding as though the
        card were empty is exactly the bug this guards against.
        """
        for name, consumer in self.consumers.items():
            known = {h.residency.key for h in self._held.values() if h.residency.consumer == name}
            for r in await consumer.residents():
                if r.key in known:
                    continue
                self._adopt_clock -= 1.0
                self._held[(name, r.key)] = _Held(
                    residency=Residency(
                        consumer=name,
                        key=r.key,
                        footprint_mb=r.footprint_mb,
                        pinned=False,
                    ),
                    last_used=self._adopt_clock,
                )

    def _lru_victim(self) -> tuple[str, str] | None:
        candidates = [k for k, h in self._held.items() if not h.residency.pinned]
        if not candidates:
            return None
        return min(candidates, key=lambda k: self._held[k].last_used)

    async def _evict(self, ident: tuple[str, str], *, reason: str) -> None:
        consumer, key = ident
        started = time.monotonic()
        # `key` is already in the consumer's key space (see _claim_locked), so
        # the confirmation compares like with like. Comparing a config ref
        # against a tagged name would report an eviction that never happened.
        await self.consumers[consumer].release(key)
        await self._confirm_gone(consumer, key)
        self._held.pop(ident, None)
        self.swaps.append(
            SwapEvent(evicted=key, reason=reason, duration_s=time.monotonic() - started)
        )

    async def _confirm_gone(self, consumer: str, key: str) -> None:
        """Confirm the release actually happened. Never assume it did."""
        still = [r.key for r in await self.consumers[consumer].residents()]
        if key in still:
            raise EvictionError(
                f"consumer '{consumer}' still reports '{key}' as resident after release; "
                "refusing to proceed rather than risk spilling to system RAM"
            )

    async def _release_everything(self, *, reason: str) -> None:
        for name, consumer in self.consumers.items():
            started = time.monotonic()
            await consumer.release_all()
            remaining = await consumer.residents()
            if remaining:
                raise EvictionError(
                    f"consumer '{name}' still holds "
                    f"{[r.key for r in remaining]} after release_all; "
                    "refusing to grant an exclusive claim"
                )
            self.swaps.append(
                SwapEvent(
                    evicted=f"{name}:*",
                    reason=reason,
                    duration_s=time.monotonic() - started,
                )
            )
        self._held.clear()

    async def restore_pinned(self, specs: list[tuple[str, str, int]]) -> None:
        """Reload pinned models after an exclusive claim released them.

        Failures here are raised, not swallowed: a harness that silently came
        back without its pinned fast model would make every later request
        mysteriously slower with no explanation.
        """
        async with self._lock:
            for consumer, key, need_mb in specs:
                await self._claim_locked(
                    consumer, key, need_mb, pinned=True, exclusive=False
                )
