from dataclasses import dataclass, field
from pathlib import Path
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
    footprint_mb: int
    kv_mb: int
    pinned: bool = False

    @property
    def need_mb(self) -> int:
        """What the arbiter is asked to reserve: the model plus its KV cache."""
        return self.footprint_mb + self.kv_mb


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


def _footprint_mb(cfg: Config, model_name: str) -> int:
    spec = cfg.models[model_name]
    return spec.footprint_mb if spec.footprint_mb is not None else DEFAULT_FOOTPRINT_MB


def _kv_mb(cfg: Config, params: dict[str, Any]) -> int:
    num_ctx = int(params.get("num_ctx", DEFAULT_NUM_CTX))
    return estimate_kv_mb(num_ctx, cfg.kv_mb_per_1k_ctx)


def resolve_prompt(
    prompt: str | None, base_dir: Path | None, *, task_name: str
) -> str | None:
    """Read the prompt file a `prompt:` key names, relative to `base_dir`.

    `prompt:` is a path, not prompt text (`prompt: prompts/summarize.md`), and
    it is resolved against the directory holding the config file rather than the
    process working directory -- the container mounts the config at
    /config/onecard.yaml while the working directory is /app.

    When `base_dir` is None the value is used verbatim as literal prompt text.
    That is the programmatic entry point for callers with no config file on
    disk; every shipped surface passes a base_dir.
    """
    if prompt is None or base_dir is None:
        return prompt
    path = (base_dir / prompt).resolve()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"task '{task_name}': prompt file '{path}' could not be read: {exc}"
        ) from exc


def build_plan(cfg: Config, task_name: str, base_dir: Path | None = None) -> ExecPlan:
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
                prompt=resolve_prompt(task.prompt, base_dir, task_name=task_name),
                params=dict(task.params),
                footprint_mb=_footprint_mb(cfg, task.model),
                kv_mb=_kv_mb(cfg, task.params),
                pinned=cfg.models[task.model].residency == "pinned",
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
                    prompt=resolve_prompt(raw.prompt, base_dir, task_name=task_name),
                    params=params,
                    footprint_mb=_footprint_mb(cfg, raw.model),
                    kv_mb=_kv_mb(cfg, params),
                    pinned=cfg.models[raw.model].residency == "pinned",
                )
            )

    return ExecPlan(task=task_name, steps=steps, exclusive=task.exclusive)
