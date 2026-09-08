"""The arbiter must reconcile with the card before it trusts its own bookkeeping.

Ollama keeps a model resident for its keep_alive after the process that loaded
it exits. A fresh Arbiter that believed its own empty `_held` would load on top
of that model and spill to system RAM -- the exact failure this project exists
to prevent, and one that looks like a hang rather than an error.
"""

import pytest

from onecard.errors import BackendError, BudgetError
from onecard.gpu.arbiter import Arbiter
from onecard.gpu.consumer import Residency
from onecard.gpu.fake import FakeConsumer


def arb(budget: int, **footprints: int) -> tuple[Arbiter, FakeConsumer]:
    c = FakeConsumer("ollama", dict(footprints))
    return Arbiter(budget_mb=budget, consumers={"ollama": c}), c


async def preload(consumer: FakeConsumer, key: str) -> None:
    """Load a model behind the arbiter's back, as a previous process would."""
    await consumer.load(key, pinned=False, footprint_hint_mb=None)


async def test_claim_evicts_a_model_this_process_never_loaded():
    a, c = arb(1000, stale=700, wanted=600)
    await preload(c, "stale")

    await a.claim("ollama", "wanted", need_mb=600)

    assert {r.key for r in await a.residents()} == {"wanted"}
    assert await a.used_mb() == 600, "the stale model's 700MB must not still be counted"
    assert a.swaps[0].evicted == "stale"


async def test_adopted_model_is_evicted_before_one_this_process_loaded():
    a, c = arb(1000, mine=400, stale=400, wanted=400)
    await preload(c, "stale")
    await a.claim("ollama", "mine", need_mb=400)

    await a.claim("ollama", "wanted", need_mb=400)

    assert {r.key for r in await a.residents()} == {"mine", "wanted"}
    assert a.swaps[0].evicted == "stale"


async def test_claim_that_cannot_fit_even_after_eviction_raises_and_loads_nothing():
    a, c = arb(1000, stale=400, huge=900)
    await preload(c, "stale")
    await a.claim("ollama", "pin", need_mb=400, pinned=True)

    with pytest.raises(BudgetError, match="cannot fit"):
        await a.claim("ollama", "huge", need_mb=900)

    assert c.load_count["huge"] == 0
    assert "huge" not in {r.key for r in await a.residents()}


async def test_reconciliation_does_not_downgrade_a_pinned_entry():
    a, c = arb(2000, p=400, stale=400, other=400)
    await a.claim("ollama", "p", need_mb=400, pinned=True)
    await preload(c, "stale")

    await a.claim("ollama", "other", need_mb=400)

    held = {r.key: r for r in await a.residents()}
    assert held["p"].pinned is True
    assert c.load_count["p"] == 1, "the pinned model must not have been reloaded"


async def test_an_undeclared_model_on_the_card_still_counts_against_the_budget():
    """The budget is about the card, not about our config."""
    a, c = arb(1000, mystery=800, wanted=400)
    await preload(c, "mystery")

    await a.claim("ollama", "wanted", need_mb=400)

    assert a.swaps[0].evicted == "mystery"
    assert {r.key for r in await a.residents()} == {"wanted"}


async def test_an_unreachable_consumer_propagates_instead_of_looking_empty():
    class Unreachable(FakeConsumer):
        async def residents(self) -> list[Residency]:
            raise BackendError("ollama unreachable at http://localhost:11434")

    c = Unreachable("ollama", {"a": 400})
    a = Arbiter(budget_mb=1000, consumers={"ollama": c})

    with pytest.raises(BackendError, match="unreachable"):
        await a.claim("ollama", "a", need_mb=400)

    assert c.load_count["a"] == 0
