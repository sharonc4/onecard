import pytest

from onecard.errors import BudgetError
from onecard.gpu.arbiter import Arbiter
from onecard.gpu.fake import FakeConsumer


def arb(budget=1000, **footprints) -> tuple[Arbiter, FakeConsumer]:
    c = FakeConsumer("ollama", dict(footprints))
    return Arbiter(budget_mb=budget, consumers={"ollama": c}), c


async def test_claim_that_fits_loads_without_eviction():
    a, _c = arb(budget=1000, a=400)
    await a.claim("ollama", "a", need_mb=400)
    assert [r.key for r in await a.residents()] == ["a"]
    assert a.swaps == []


async def test_second_claim_that_fits_keeps_both():
    a, _c = arb(budget=1000, a=400, b=400)
    await a.claim("ollama", "a", need_mb=400)
    await a.claim("ollama", "b", need_mb=400)
    assert {r.key for r in await a.residents()} == {"a", "b"}


async def test_claim_evicts_least_recently_used():
    a, _c = arb(budget=1000, a=400, b=400, d=400)
    await a.claim("ollama", "a", need_mb=400)
    await a.claim("ollama", "b", need_mb=400)
    await a.claim("ollama", "a", need_mb=400)   # touch 'a' so 'b' is now LRU
    await a.claim("ollama", "d", need_mb=400)
    assert {r.key for r in await a.residents()} == {"a", "d"}
    assert a.swaps[0].evicted == "b"


async def test_pinned_models_are_not_evicted_by_ordinary_claims():
    a, _c = arb(budget=1000, p=400, x=400, y=400)
    await a.claim("ollama", "p", need_mb=400, pinned=True)
    await a.claim("ollama", "x", need_mb=400)
    await a.claim("ollama", "y", need_mb=400)
    assert "p" in {r.key for r in await a.residents()}
    assert a.swaps[0].evicted == "x"


async def test_claim_larger_than_budget_raises_and_loads_nothing():
    a, _c = arb(budget=1000, huge=2000)
    with pytest.raises(BudgetError, match="exceeds the budget"):
        await a.claim("ollama", "huge", need_mb=2000)
    assert await a.residents() == []


async def test_claim_that_cannot_fit_around_pinned_raises():
    a, _c = arb(budget=1000, p=800, big=400)
    await a.claim("ollama", "p", need_mb=800, pinned=True)
    with pytest.raises(BudgetError, match="cannot fit"):
        await a.claim("ollama", "big", need_mb=400)


async def test_reclaiming_a_resident_model_does_not_reload_it():
    a, c = arb(budget=1000, a=400)
    await a.claim("ollama", "a", need_mb=400)
    await a.claim("ollama", "a", need_mb=400)
    assert c.load_count["a"] == 1


async def test_unknown_consumer_raises():
    a, _ = arb()
    with pytest.raises(KeyError):
        await a.claim("nope", "a", need_mb=1)
