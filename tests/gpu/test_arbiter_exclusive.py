import asyncio
import contextvars

import pytest

from onecard.errors import EvictionError
from onecard.gpu.arbiter import Arbiter
from onecard.gpu.consumer import Residency
from onecard.gpu.fake import FakeConsumer


def two_consumer_arb(budget=8000):
    ollama = FakeConsumer("ollama", {"small": 900, "big": 5000})
    comfy = FakeConsumer("comfyui", {"sd15": 4000, "sd15b": 4000})
    arb = Arbiter(budget_mb=budget, consumers={"ollama": ollama, "comfyui": comfy})
    return arb, ollama, comfy


async def test_exclusive_claim_evicts_everything_including_pinned():
    a, ollama, _comfy = two_consumer_arb()
    await a.claim("ollama", "small", need_mb=900, pinned=True)
    await a.claim("ollama", "big", need_mb=5000)
    await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)

    assert [r.key for r in await a.residents()] == ["sd15"]
    assert await ollama.residents() == []


async def test_exclusive_claim_fails_loudly_if_a_consumer_will_not_release():
    liar = FakeConsumer("ollama", {"stuck": 5000}, honest=False)
    comfy = FakeConsumer("comfyui", {"sd15": 4000})
    a = Arbiter(budget_mb=8000, consumers={"ollama": liar, "comfyui": comfy})
    await a.claim("ollama", "stuck", need_mb=5000)

    with pytest.raises(EvictionError, match="still holds"):
        await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)

    assert await comfy.residents() == [], "must not load while the card is still occupied"


async def test_restore_pinned_reloads_after_exclusive_work():
    a, _ollama, _comfy = two_consumer_arb()
    await a.claim("ollama", "small", need_mb=900, pinned=True)
    await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)

    await a.restore_pinned([("ollama", "small", 900)])
    assert "small" in {r.key for r in await a.residents()}


async def test_restore_pinned_is_idempotent():
    a, ollama, _comfy = two_consumer_arb()
    await a.claim("ollama", "small", need_mb=900, pinned=True)
    await a.restore_pinned([("ollama", "small", 900)])
    assert ollama.load_count["small"] == 1


# Task-local label for the claim currently in flight. Each asyncio Task gets
# its own copy of the contextvars context at creation time, so setting this
# inside one gathered coroutine cannot leak into the other's — unlike a
# label stored as a mutable attribute on the shared consumer, which a
# concurrently-running task could overwrite mid-flight.
_current_label: contextvars.ContextVar[str] = contextvars.ContextVar("current_label")


class _RecordingConsumer(FakeConsumer):
    """FakeConsumer that appends labelled events to a shared log.

    Used to prove exclusive claims serialize behind the arbiter's lock: with
    a real suspension point (delay_s > 0) on load/release_all, two
    concurrently gathered exclusive claims would interleave their operations
    if the lock were not held across the whole claim.
    """

    def __init__(self, *args: object, log: list[str], **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._log = log

    async def load(
        self, key: str, *, pinned: bool, footprint_hint_mb: int | None
    ) -> Residency:
        self._log.append(f"{_current_label.get()}:load")
        return await super().load(key, pinned=pinned, footprint_hint_mb=footprint_hint_mb)

    async def release_all(self) -> None:
        self._log.append(f"{_current_label.get()}:release_all")
        await super().release_all()


async def test_concurrent_exclusive_claims_serialize():
    log: list[str] = []
    comfy = _RecordingConsumer(
        "comfyui", {"sd15": 4000, "sd15b": 4000}, delay_s=0.01, log=log
    )
    a = Arbiter(budget_mb=8000, consumers={"comfyui": comfy})
    order: list[str] = []

    async def claim(key: str, label: str) -> None:
        _current_label.set(label)
        await a.claim("comfyui", key, need_mb=4000, exclusive=True)
        order.append(f"done:{key}")

    await asyncio.gather(claim("sd15", "A"), claim("sd15b", "B"))
    assert len(order) == 2, "both claims must complete, one after the other"
    assert len([r for r in await a.residents()]) == 1, "only one may hold the card"

    # The lock must serialize entire claims: one claim's release_all+load
    # must fully finish before the other's begins. Interleaving such as
    # [A:release_all, B:release_all, A:load, B:load] must fail here.
    assert log in (
        ["A:release_all", "A:load", "B:release_all", "B:load"],
        ["B:release_all", "B:load", "A:release_all", "A:load"],
    ), f"exclusive claims interleaved: {log}"


async def test_swap_events_record_exclusive_reason():
    a, _ollama, _comfy = two_consumer_arb()
    await a.claim("ollama", "big", need_mb=5000)
    await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)
    assert any("exclusive claim" in s.reason for s in a.swaps)
