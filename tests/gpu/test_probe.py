"""Tests for the VRAM probe and the arbiter's use of it.

The scenarios here are not invented. They are replayed from measurements taken
on an RTX 2070 SUPER (8GB):

- idle VRAM was 662MiB headless, 894MiB with one display on the motherboard,
  and 1630MiB with both displays driven by the card
- Ollama's container held ~6GB while a model was loaded, and a
  `keep_alive: 0` release did not give it back
- a 10.08GB peak on the 8GB card did not fail; it spilled to system RAM and
  took 103s where a 5.28GB peak took 2.3s
"""

import pytest

from onecard.errors import BudgetError
from onecard.gpu.arbiter import Arbiter
from onecard.gpu.fake import FakeConsumer
from onecard.gpu.probe import (
    NvidiaSmiProbe,
    StaticVramProbe,
    VramProbe,
    VramReading,
    parse_nvidia_smi,
)

CARD_TOTAL_MB = 8192


def test_parses_nvidia_smi_output():
    reading = parse_nvidia_smi("8192, 1630, 6562\n")
    assert reading == VramReading(total_mb=8192, used_mb=1630, free_mb=6562)


def test_parses_first_gpu_only():
    reading = parse_nvidia_smi("8192, 1630, 6562\n24576, 1000, 23576\n")
    assert reading is not None
    assert reading.total_mb == 8192


def test_unparseable_output_is_unknown_not_zero():
    """A garbled reading must never look like an empty card."""
    assert parse_nvidia_smi("no devices were found") is None
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("8192, not-a-number, 6562") is None


def test_missing_nvidia_smi_reads_as_unknown():
    assert NvidiaSmiProbe(executable="definitely-not-a-real-binary-xyz").read() is None


def test_static_probe_satisfies_the_protocol():
    assert isinstance(StaticVramProbe(None), VramProbe)
    assert isinstance(NvidiaSmiProbe(), VramProbe)


def arbiter_with(free_mb: int | None, budget_mb: int = 6800) -> Arbiter:
    consumer = FakeConsumer("ollama", {"reasoner": 5000, "coder": 4700})
    reading = (
        None
        if free_mb is None
        else VramReading(
            total_mb=CARD_TOTAL_MB, used_mb=CARD_TOTAL_MB - free_mb, free_mb=free_mb
        )
    )
    return Arbiter(
        budget_mb=budget_mb,
        consumers={"ollama": consumer},
        probe=StaticVramProbe(reading),
    )


async def test_claim_is_refused_when_the_card_is_held_by_something_else():
    """The 916s/step case: Ollama's container holds ~6GB it will not release.

    onecard's own bookkeeping says the card is empty and the budget has room,
    so without the probe this claim is granted and silently spills.
    """
    a = arbiter_with(free_mb=1500)
    with pytest.raises(BudgetError, match="held by something"):
        await a.claim("ollama", "reasoner", need_mb=5000)
    assert await a.residents() == [], "must not load when the card has no room"


async def test_the_refusal_names_the_numbers_a_user_needs():
    a = arbiter_with(free_mb=1500)
    with pytest.raises(BudgetError) as excinfo:
        await a.claim("ollama", "reasoner", need_mb=5000)
    message = str(excinfo.value)
    assert "5000MB" in message and "1500MB" in message and "8192MB" in message
    assert "spill" in message


async def test_claim_proceeds_when_the_card_genuinely_has_room():
    a = arbiter_with(free_mb=7298)  # measured: 8192 total, 894 idle, one display
    await a.claim("ollama", "reasoner", need_mb=5000)
    assert [r.key for r in await a.residents()] == ["reasoner"]


async def test_a_probe_that_cannot_read_does_not_block_the_claim():
    """Unknown must not mean refuse — a CPU-only contributor has no nvidia-smi."""
    a = arbiter_with(free_mb=None)
    await a.claim("ollama", "reasoner", need_mb=5000)
    assert [r.key for r in await a.residents()] == ["reasoner"]


async def test_no_probe_at_all_behaves_exactly_as_before():
    consumer = FakeConsumer("ollama", {"reasoner": 5000})
    a = Arbiter(budget_mb=6800, consumers={"ollama": consumer})
    await a.claim("ollama", "reasoner", need_mb=5000)
    assert [r.key for r in await a.residents()] == ["reasoner"]


async def test_budget_still_binds_even_when_the_card_looks_empty():
    """The probe is an extra gate, never a licence to exceed the budget."""
    a = arbiter_with(free_mb=8192, budget_mb=4000)
    with pytest.raises(BudgetError, match="exceeds the budget"):
        await a.claim("ollama", "reasoner", need_mb=5000)


async def test_reproduces_the_measured_quantization_ceiling():
    """The measured boundary on this card, headless at 662MB idle: the Q4_K_S
    Flux UNet (6.33GB) fits and Q5_K_S (7.71GB) does not — only ~7.3GB is free
    and activations need room on top of the weights."""
    headless_free = CARD_TOTAL_MB - 662  # 7530MB
    q4_k_s_mb = 6482  # 6.33 GB — ran at 227.4s/frame
    q5_k_s_mb = 7895  # 7.71 GB — never fit

    fits = arbiter_with(free_mb=headless_free, budget_mb=CARD_TOTAL_MB)
    await fits.claim("ollama", "reasoner", need_mb=q4_k_s_mb)
    assert [r.key for r in await fits.residents()] == ["reasoner"]

    too_big = arbiter_with(free_mb=headless_free, budget_mb=CARD_TOTAL_MB)
    with pytest.raises(BudgetError, match="only 7530MB free"):
        await too_big.claim("ollama", "reasoner", need_mb=q5_k_s_mb)


async def test_displays_cost_the_measured_amount_of_headroom():
    """Driving both monitors from the card took idle VRAM from 662MB to
    1630MB — 968MB that a claim can no longer use."""
    headless = arbiter_with(free_mb=CARD_TOTAL_MB - 662, budget_mb=CARD_TOTAL_MB)
    with_displays = arbiter_with(free_mb=CARD_TOTAL_MB - 1630, budget_mb=CARD_TOTAL_MB)

    borderline = CARD_TOTAL_MB - 1000  # fits headless, not with both displays
    await headless.claim("ollama", "reasoner", need_mb=borderline)
    with pytest.raises(BudgetError):
        await with_displays.claim("ollama", "reasoner", need_mb=borderline)
