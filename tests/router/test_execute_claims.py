"""What the executor actually asks the arbiter for: prompts, pinning, footprints."""

import textwrap
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from onecard.config.loader import load_config
from onecard.gpu.arbiter import Arbiter
from onecard.gpu.consumer import Residency
from onecard.gpu.fake import FakeConsumer
from onecard.router.execute import Executor
from onecard.router.plan import build_plan
from onecard.store.footprints import FootprintStore

CONFIG = textwrap.dedent(
    """
    vram_budget_mb: 8000
    models:
      fast:  { ref: "qwen-fast", residency: pinned, footprint_mb: 900 }
      coder: { ref: "qwen-coder", footprint_mb: 4000 }
    tasks:
      code:      { model: coder, prompt: prompts/code.md, params: { num_ctx: 1024 } }
      summarize: { model: fast, params: { num_ctx: 1024 } }
    """
)


class RecordingArbiter(Arbiter):
    """Arbiter that remembers every claim it was asked to grant."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.claims: list[tuple[str, int, bool]] = []

    async def claim(
        self,
        consumer: str,
        key: str,
        need_mb: int,
        *,
        pinned: bool = False,
        exclusive: bool = False,
    ) -> Residency:
        self.claims.append((key, need_mb, pinned))
        return await super().claim(
            consumer, key, need_mb, pinned=pinned, exclusive=exclusive
        )


def project(tmp_path: Path, prompt_body: str = "Explain this code.") -> Path:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "code.md").write_text(prompt_body, encoding="utf-8")
    config = tmp_path / "onecard.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    return config


def harness(tmp_path: Path) -> tuple[Executor, RecordingArbiter, list[str], FootprintStore]:
    consumer = FakeConsumer("ollama", {"qwen-coder": 4000, "qwen-fast": 900})
    arbiter = RecordingArbiter(budget_mb=8000, consumers={"ollama": consumer})
    prompts: list[str] = []
    store = FootprintStore(tmp_path / "f.db")

    async def chat_fn(model_ref: str, prompt: str, params: dict[str, Any]) -> AsyncIterator[str]:
        prompts.append(prompt)
        yield "ok"

    return (
        Executor(arbiter=arbiter, chat_fn=chat_fn, store=store, on_event=lambda _m: None),
        arbiter,
        prompts,
        store,
    )


async def test_prompt_file_contents_reach_the_model_not_the_path(tmp_path: Path):
    config = project(tmp_path, "You are a code reviewer. Be terse.")
    cfg = load_config(config)
    ex, _arb, prompts, _store = harness(tmp_path)

    [c async for c in ex.run(build_plan(cfg, "code", config.parent), "def f(): pass")]

    assert prompts == ["You are a code reviewer. Be terse.\n\ndef f(): pass"]
    assert "prompts/code.md" not in prompts[0]


async def test_a_pinned_model_is_claimed_as_pinned_and_survives_a_later_claim(tmp_path: Path):
    config = project(tmp_path)
    cfg = load_config(config)
    ex, arb, _prompts, _store = harness(tmp_path)

    [c async for c in ex.run(build_plan(cfg, "summarize", config.parent), "text")]
    assert arb.claims == [("qwen-fast", 900 + 64, True)]

    [c async for c in ex.run(build_plan(cfg, "code", config.parent), "code")]

    resident = {r.key: r for r in await arb.residents()}
    assert resident["qwen-fast"].pinned is True
    assert arb.swaps == [], "a pinned model must not be evicted for a later claim"


async def test_a_measured_footprint_is_preferred_over_the_declared_estimate(tmp_path: Path):
    config = project(tmp_path)
    cfg = load_config(config)
    ex, arb, _prompts, store = harness(tmp_path)
    store.record("ollama", "qwen-coder", 3100)  # measured on a previous run

    [c async for c in ex.run(build_plan(cfg, "code", config.parent), "code")]

    assert arb.claims == [("qwen-coder", 3100 + 64, False)]
