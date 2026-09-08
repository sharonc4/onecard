from pathlib import Path
from typing import Any

from onecard.config.schema import Config, TaskSpec
from onecard.errors import ConfigError
from onecard.gpu.consumer import DEFAULT_FOOTPRINT_MB
from onecard.gpu.estimate import estimate_kv_mb

DEFAULT_NUM_CTX = 2048


def _declared_sources(task: TaskSpec) -> int:
    return sum(x is not None for x in (task.model, task.steps, task.workflow))


def _fit_check_pairs(task: TaskSpec) -> list[tuple[str, dict[str, Any]]]:
    """Every (model, effective-params) pair a task can execute.

    A task with `model` yields one pair with the task's own params. A task with
    `steps` yields one pair per step, with step params overriding task params —
    this must match the merge `build_plan` performs. A `workflow` task involves
    no language model and yields nothing.
    """
    if task.model is not None:
        return [(task.model, task.params)]
    if task.steps is not None:
        return [(step.model, {**task.params, **step.params}) for step in task.steps]
    return []


def _prompt_paths(task: TaskSpec) -> list[str]:
    return [p for p in [task.prompt, *[s.prompt for s in task.steps or []]] if p is not None]


def validate_config(cfg: Config, base_dir: Path | None = None) -> list[str]:
    """Raise ConfigError on anything fatal; return non-fatal warnings.

    `base_dir` is the directory holding the config file, against which every
    `prompt:` path is resolved. When it is None the prompt-file check is
    skipped rather than guessed at, because there is nothing to resolve
    against -- see `router.plan.resolve_prompt`.
    """
    warnings: list[str] = []

    for name, task in cfg.tasks.items():
        if _declared_sources(task) != 1:
            raise ConfigError(
                f"task '{name}': declare exactly one of 'model', 'steps', or 'workflow'"
            )
        refs = [task.model] if task.model else [s.model for s in (task.steps or [])]
        for ref in refs:
            if ref not in cfg.models:
                raise ConfigError(f"task '{name}': unknown model '{ref}'")
        if base_dir is not None:
            for prompt in _prompt_paths(task):
                path = (base_dir / prompt).resolve()
                if not path.is_file():
                    raise ConfigError(
                        f"task '{name}': prompt file '{path}' does not exist "
                        f"(prompt: {prompt})"
                    )

    pinned_mb = 0
    for name, spec in cfg.models.items():
        if spec.footprint_mb is None:
            warnings.append(
                f"model '{name}': no footprint declared; it will be measured on first load"
            )
        elif spec.residency == "pinned":
            pinned_mb += spec.footprint_mb

    if pinned_mb > cfg.vram_budget_mb:
        raise ConfigError(
            f"pinned models require {pinned_mb}MB but the budget is {cfg.vram_budget_mb}MB"
        )

    for name, task in cfg.tasks.items():
        for model_ref, effective_params in _fit_check_pairs(task):
            spec = cfg.models[model_ref]
            if spec.residency == "pinned":
                continue
            # An undeclared footprint is not a free pass: use the same fallback
            # `router.plan` uses at runtime, or validate would say "ok" to a
            # config that raises BudgetError on its first request.
            footprint = (
                spec.footprint_mb if spec.footprint_mb is not None else DEFAULT_FOOTPRINT_MB
            )
            num_ctx = int(effective_params.get("num_ctx", DEFAULT_NUM_CTX))
            need = footprint + estimate_kv_mb(num_ctx, cfg.kv_mb_per_1k_ctx)
            if need + pinned_mb > cfg.vram_budget_mb:
                where = (
                    f"task '{name}'"
                    if task.model is not None
                    else f"task '{name}' step model '{model_ref}'"
                )
                raise ConfigError(
                    f"{where} can never fit: needs {need}MB plus {pinned_mb}MB pinned, "
                    f"budget is {cfg.vram_budget_mb}MB"
                )

    return warnings
