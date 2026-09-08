from onecard.config.schema import Config, TaskSpec
from onecard.errors import ConfigError
from onecard.gpu.estimate import estimate_kv_mb

DEFAULT_NUM_CTX = 2048


def _declared_sources(task: TaskSpec) -> int:
    return sum(x is not None for x in (task.model, task.steps, task.workflow))


def validate_config(cfg: Config) -> list[str]:
    """Raise ConfigError on anything fatal; return non-fatal warnings."""
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
        if task.model is None:
            continue
        spec = cfg.models[task.model]
        if spec.footprint_mb is None or spec.residency == "pinned":
            continue
        num_ctx = int(task.params.get("num_ctx", DEFAULT_NUM_CTX))
        need = spec.footprint_mb + estimate_kv_mb(num_ctx, cfg.kv_mb_per_1k_ctx)
        if need + pinned_mb > cfg.vram_budget_mb:
            raise ConfigError(
                f"task '{name}' can never fit: needs {need}MB plus {pinned_mb}MB pinned, "
                f"budget is {cfg.vram_budget_mb}MB"
            )

    return warnings
