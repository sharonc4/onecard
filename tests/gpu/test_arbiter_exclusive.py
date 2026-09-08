import asyncio

import pytest

from onecard.errors import EvictionError
from onecard.gpu.arbiter import Arbiter
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


async def test_concurrent_exclusive_claims_serialize():
    a, _ollama, _comfy = two_consumer_arb()
    order: list[str] = []

    async def claim(key: str) -> None:
        await a.claim("comfyui", key, need_mb=4000, exclusive=True)
        order.append(f"done:{key}")

    await asyncio.gather(claim("sd15"), claim("sd15b"))
    assert len(order) == 2, "both claims must complete, one after the other"
    assert len([r for r in await a.residents()]) == 1, "only one may hold the card"


async def test_swap_events_record_exclusive_reason():
    a, _ollama, _comfy = two_consumer_arb()
    await a.claim("ollama", "big", need_mb=5000)
    await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)
    assert any("exclusive claim" in s.reason for s in a.swaps)
