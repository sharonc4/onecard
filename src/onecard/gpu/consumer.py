from dataclasses import dataclass
from typing import Protocol, runtime_checkable

DEFAULT_FOOTPRINT_MB = 4500


@dataclass(frozen=True)
class Residency:
    consumer: str
    key: str
    footprint_mb: int
    pinned: bool = False


@runtime_checkable
class GpuConsumer(Protocol):
    """Anything that can hold the GPU. Ollama and ComfyUI both implement this."""

    name: str

    def key_for(self, ref: str) -> str:
        """This consumer's own key for a config ref.

        The arbiter keys its bookkeeping by this so that its key space and the
        backend's are the same one: Ollama reports "nomic-embed-text:latest"
        for a ref written "nomic-embed-text", and an arbiter comparing the two
        forms would confirm evictions that never happened.
        """

    async def residents(self) -> list[Residency]:
        """Ground truth: what this consumer currently holds."""

    async def load(
        self, key: str, *, pinned: bool, footprint_hint_mb: int | None
    ) -> Residency: ...

    async def release(self, key: str) -> None: ...

    async def release_all(self) -> None: ...
