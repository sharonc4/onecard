import asyncio
from collections import defaultdict

from onecard.gpu.consumer import DEFAULT_FOOTPRINT_MB, Residency


class FakeConsumer:
    """In-memory GpuConsumer for tests.

    honest=False models a backend that acknowledges a release but has not
    actually freed the memory — the case the arbiter must detect.

    delay_s, when non-zero, makes load/release/release_all await
    asyncio.sleep(delay_s) before doing their work, giving tests a real
    suspension point to prove interleaving (or its absence) across
    concurrent coroutines. Defaults to 0.0, which preserves the previous
    behavior exactly (no sleep, not even asyncio.sleep(0)).
    """

    def __init__(
        self,
        name: str,
        footprints: dict[str, int],
        honest: bool = True,
        delay_s: float = 0.0,
    ) -> None:
        self.name = name
        self.footprints = footprints
        self.honest = honest
        self.delay_s = delay_s
        self._resident: dict[str, Residency] = {}
        self.load_count: dict[str, int] = defaultdict(int)
        self.release_all_count = 0

    def key_for(self, ref: str) -> str:
        return ref

    async def residents(self) -> list[Residency]:
        return list(self._resident.values())

    async def load(
        self, key: str, *, pinned: bool, footprint_hint_mb: int | None
    ) -> Residency:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        mb = self.footprints.get(key, footprint_hint_mb or DEFAULT_FOOTPRINT_MB)
        r = Residency(consumer=self.name, key=key, footprint_mb=mb, pinned=pinned)
        self._resident[key] = r
        self.load_count[key] += 1
        return r

    async def release(self, key: str) -> None:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.honest:
            self._resident.pop(key, None)

    async def release_all(self) -> None:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        self.release_all_count += 1
        if self.honest:
            self._resident.clear()
