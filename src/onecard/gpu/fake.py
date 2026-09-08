from collections import defaultdict

from onecard.gpu.consumer import DEFAULT_FOOTPRINT_MB, Residency


class FakeConsumer:
    """In-memory GpuConsumer for tests.

    honest=False models a backend that acknowledges a release but has not
    actually freed the memory — the case the arbiter must detect.
    """

    def __init__(self, name: str, footprints: dict[str, int], honest: bool = True) -> None:
        self.name = name
        self.footprints = footprints
        self.honest = honest
        self._resident: dict[str, Residency] = {}
        self.load_count: dict[str, int] = defaultdict(int)
        self.release_all_count = 0

    async def residents(self) -> list[Residency]:
        return list(self._resident.values())

    async def load(
        self, key: str, *, pinned: bool, footprint_hint_mb: int | None
    ) -> Residency:
        mb = self.footprints.get(key, footprint_hint_mb or DEFAULT_FOOTPRINT_MB)
        r = Residency(consumer=self.name, key=key, footprint_mb=mb, pinned=pinned)
        self._resident[key] = r
        self.load_count[key] += 1
        return r

    async def release(self, key: str) -> None:
        if self.honest:
            self._resident.pop(key, None)

    async def release_all(self) -> None:
        self.release_all_count += 1
        if self.honest:
            self._resident.clear()
