from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from onecard.gpu.arbiter import Arbiter
from onecard.router.plan import ExecPlan
from onecard.store.footprints import FootprintStore


class ChatFn(Protocol):
    def __call__(
        self, model_ref: str, prompt: str, params: dict[str, Any]
    ) -> AsyncIterator[str]: ...


def _noop(_: str) -> None:
    return None


class Executor:
    """Runs an ExecPlan: claim the GPU per step, stream the reply, chain outputs."""

    def __init__(
        self,
        arbiter: Arbiter,
        chat_fn: ChatFn,
        store: FootprintStore,
        on_event: Callable[[str], None] = _noop,
    ) -> None:
        self.arbiter = arbiter
        self.chat_fn = chat_fn
        self.store = store
        self.on_event = on_event

    async def run(self, plan: ExecPlan, user_input: str) -> AsyncIterator[str]:
        if plan.workflow is not None:
            raise NotImplementedError(
                "image tasks require the ComfyUI consumer, which is not part of the core build"
            )

        current = user_input
        loaded: str | None = None
        last = plan.steps[-1] if plan.steps else None

        for step in plan.steps:
            if loaded != step.model_name:
                seen_before = len(self.arbiter.swaps)
                self.on_event(f"loading {step.model_name} ({step.need_mb}MB)")
                residency = await self.arbiter.claim(
                    "ollama", step.model_ref, need_mb=step.need_mb
                )
                for swap in self.arbiter.swaps[seen_before:]:
                    self.on_event(f"evicting {swap.evicted} ({swap.duration_s:.1f}s)")
                self.store.record("ollama", step.model_ref, residency.footprint_mb)
                loaded = step.model_name

            prompt = f"{step.prompt}\n\n{current}" if step.prompt else current
            collected: list[str] = []
            async for chunk in self.chat_fn(step.model_ref, prompt, step.params):
                collected.append(chunk)
                if step is last:
                    yield chunk
            current = "".join(collected)
