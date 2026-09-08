from onecard.gpu.fake import FakeConsumer


async def test_load_then_residents_reports_it():
    c = FakeConsumer("fake", {"a": 100})
    r = await c.load("a", pinned=False, footprint_hint_mb=None)
    assert r.key == "a"
    assert r.footprint_mb == 100
    assert [x.key for x in await c.residents()] == ["a"]


async def test_release_removes_it():
    c = FakeConsumer("fake", {"a": 100})
    await c.load("a", pinned=False, footprint_hint_mb=None)
    await c.release("a")
    assert await c.residents() == []


async def test_release_all_clears_even_pinned():
    c = FakeConsumer("fake", {"a": 100, "b": 200})
    await c.load("a", pinned=True, footprint_hint_mb=None)
    await c.load("b", pinned=False, footprint_hint_mb=None)
    await c.release_all()
    assert await c.residents() == []


async def test_dishonest_consumer_keeps_reporting_after_release():
    """Simulates a backend that acknowledges eviction but has not actually freed VRAM."""
    c = FakeConsumer("fake", {"a": 100}, honest=False)
    await c.load("a", pinned=False, footprint_hint_mb=None)
    await c.release("a")
    assert [x.key for x in await c.residents()] == ["a"]


async def test_unknown_key_uses_hint():
    c = FakeConsumer("fake", {})
    r = await c.load("ghost", pinned=False, footprint_hint_mb=4321)
    assert r.footprint_mb == 4321


async def test_load_counts_are_recorded():
    c = FakeConsumer("fake", {"a": 100})
    await c.load("a", pinned=False, footprint_hint_mb=None)
    await c.release("a")
    await c.load("a", pinned=False, footprint_hint_mb=None)
    assert c.load_count["a"] == 2
