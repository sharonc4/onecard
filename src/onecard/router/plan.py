from dataclasses import dataclass, field
from typing import Any

from onecard.config.schema import Config, TaskSpec
from onecard.errors import ConfigError
from onecard.gpu.consumer import DEFAULT_FOOTPRINT_MB
from onecard.gpu.estimate import estimate_kv_mb

DEFAULT_NUM_CTX = 2048


@dataclass(frozen=True)
class ExecStep:
    model_name: str
    model_ref: str
    prompt: str | None
    params: dict[str, Any]
    need_mb: int


@dataclass(frozen=True)
class ExecPlan:
    task: str
    steps: list[ExecStep] = field(default_factory=list)
    exclusive: bool = False
    workflow: str | None = None

    @property
    def load_sequence(self) -> list[str]:
        """Distinct models in load order; consecutive duplicates collapsed.

        This is what makes a three-step job swap once instead of three times.
        """
        seq: list[str] = []
        for step in self.steps:
            if not seq or seq[-1] != step.model_name:
                seq.append(step.model_name)
        return seq


def _need_mb(cfg: Config, model_name: str, params: dict[str, Any]) -> int:
    spec = cfg.models[model_name]
    footprint = (
        spec.footprint_mb if spec.footprint_mb is not None else DEFAULT_FOOTPRINT_MB
    )
    num_ctx = int(params.get("num_ctx", DEFAULT_NUM_CTX))
    return footprint + estimate_kv_mb(num_ctx, cfg.kv_mb_per_1k_ctx)


def build_plan(cfg: Config, task_name: str) -> ExecPlan:
    task: TaskSpec | None = cfg.tasks.get(task_name)
    if task is None:
        raise ConfigError(f"unknown task '{task_name}'")

    if task.workflow is not None:
        return ExecPlan(task=task_name, steps=[], exclusive=True, workflow=task.workflow)

    if task.model is not None:
        steps = [
            ExecStep(
                model_name=task.model,
                model_ref=cfg.models[task.model].ref,
                prompt=task.prompt,
                params=dict(task.params),
                need_mb=_need_mb(cfg, task.model, task.params),
            )
        ]
    else:
        steps = []
        for raw in task.steps or []:
            params = {**task.params, **raw.params}
            steps.append(
                ExecStep(
                    model_name=raw.model,
                    model_ref=cfg.models[raw.model].ref,
                    prompt=raw.prompt,
                    params=params,
                    need_mb=_need_mb(cfg, raw.model, params),
                )
            )

    return ExecPlan(task=task_name, steps=steps, exclusive=task.exclusive)
