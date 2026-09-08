from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from onecard.gpu.arbiter import Arbiter
from onecard.gpu.fake import FakeConsumer
from onecard.router.execute import Executor
from onecard.router.plan import ExecPlan, ExecStep
from onecard.store.footprints import FootprintStore


def step(model="coder", ref="qwen-coder", prompt=None, need=4000) -> ExecStep:
    return ExecStep(model_name=model, model_ref=ref, prompt=prompt, params={}, need_mb=need)


def make(tmp_path: Path, budget=8000):
    consumer = FakeConsumer("ollama", {"qwen-coder": 4000, "qwen-fast": 900})
    arbiter = Arbiter(budget_mb=budget, consumers={"ollama": consumer})
    events: list[str] = []
    replies: list[str] = []

    async def chat_fn(model_ref: str, prompt: str, params: dict[str, Any]) -> AsyncIterator[str]:
        replies.append(f"{model_ref}:{prompt}")
        yield f"<{model_ref}>"

    ex = Executor(
        arbiter=arbiter,
        chat_fn=chat_fn,
        store=FootprintStore(tmp_path / "f.db"),
        on_event=events.append,
    )
    return ex, consumer, events, replies


async def test_single_step_streams_output(tmp_path: Path):
    ex, _, _, _ = make(tmp_path)
    out = [c async for c in ex.run(ExecPlan(task="chat", steps=[step()]), "hello")]
    assert "".join(out) == "<qwen-coder>"


async def test_user_input_is_passed_to_first_step(tmp_path: Path):
    ex, _, _, replies = make(tmp_path)
    [c async for c in ex.run(ExecPlan(task="chat", steps=[step()]), "hello")]
    assert replies[0] == "qwen-coder:hello"


async def test_pipeline_feeds_each_step_the_previous_output(tmp_path: Path):
    ex, _, _, replies = make(tmp_path)
    plan = ExecPlan(task="review", steps=[step(), step(model="fast", ref="qwen-fast", need=900)])
    [c async for c in ex.run(plan, "start")]
    assert replies == ["qwen-coder:start", "qwen-fast:<qwen-coder>"]


async def test_consecutive_same_model_steps_load_once(tmp_path: Path):
    ex, consumer, _, _ = make(tmp_path)
    plan = ExecPlan(task="review", steps=[step(), step(), step()])
    [c async for c in ex.run(plan, "x")]
    assert consumer.load_count["qwen-coder"] == 1


async def test_arbitration_events_are_reported(tmp_path: Path):
    ex, _, events, _ = make(tmp_path, budget=4500)
    plan = ExecPlan(task="review", steps=[step(), step(model="fast", ref="qwen-fast", need=900)])
    [c async for c in ex.run(plan, "x")]
    assert any("loading" in e for e in events)
    assert any("evicting" in e for e in events)


async def test_measured_footprint_is_recorded(tmp_path: Path):
    ex, _, _, _ = make(tmp_path)
    [c async for c in ex.run(ExecPlan(task="chat", steps=[step()]), "x")]
    assert ex.store.get("ollama", "qwen-coder") == 4000


async def test_workflow_plan_is_rejected_in_core(tmp_path: Path):
    """Image tasks arrive in Plan 2. Until then this fails loudly, never silently."""
    ex, _, _, _ = make(tmp_path)
    plan = ExecPlan(task="picture", steps=[], exclusive=True, workflow="w.json")
    with pytest.raises(NotImplementedError, match="image tasks"):
        [c async for c in ex.run(plan, "a cat")]
