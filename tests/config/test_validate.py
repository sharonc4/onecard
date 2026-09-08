import pytest

from onecard.config.schema import Config, ModelSpec, StepSpec, TaskSpec
from onecard.config.validate import validate_config
from onecard.errors import ConfigError


def cfg(**kw) -> Config:
    base = {
        "vram_budget_mb": 6800,
        "models": {
            "fast": ModelSpec(ref="qwen2.5:1.5b", residency="pinned", footprint_mb=900),
            "big": ModelSpec(ref="llama3.1:8b", footprint_mb=5000),
        },
        "tasks": {"chat": TaskSpec(model="big")},
    }
    base.update(kw)
    return Config(**base)


def test_valid_config_passes():
    assert validate_config(cfg()) == []


def test_task_referencing_unknown_model_fails():
    bad = cfg(tasks={"chat": TaskSpec(model="ghost")})
    with pytest.raises(ConfigError, match="unknown model 'ghost'"):
        validate_config(bad)


def test_task_with_both_model_and_steps_fails():
    bad = cfg(tasks={"chat": TaskSpec(model="big", steps=[StepSpec(model="fast")])})
    with pytest.raises(ConfigError, match="exactly one of"):
        validate_config(bad)


def test_task_with_neither_model_nor_steps_nor_workflow_fails():
    bad = cfg(tasks={"chat": TaskSpec()})
    with pytest.raises(ConfigError, match="exactly one of"):
        validate_config(bad)


def test_step_referencing_unknown_model_fails():
    bad = cfg(tasks={"chat": TaskSpec(steps=[StepSpec(model="ghost")])})
    with pytest.raises(ConfigError, match="unknown model 'ghost'"):
        validate_config(bad)


def test_pinned_models_exceeding_budget_fails():
    bad = cfg(
        vram_budget_mb=1000,
        models={
            "a": ModelSpec(ref="a", residency="pinned", footprint_mb=800),
            "b": ModelSpec(ref="b", residency="pinned", footprint_mb=800),
        },
        tasks={"chat": TaskSpec(model="a")},
    )
    with pytest.raises(ConfigError, match="pinned models require"):
        validate_config(bad)


def test_task_that_can_never_fit_fails():
    bad = cfg(
        vram_budget_mb=2000,
        models={"big": ModelSpec(ref="big", footprint_mb=5000)},
        tasks={"chat": TaskSpec(model="big")},
    )
    with pytest.raises(ConfigError, match="can never fit"):
        validate_config(bad)


def test_unmeasured_model_produces_a_warning_not_an_error():
    c = cfg(
        models={"big": ModelSpec(ref="llama3.1:8b")},
        tasks={"chat": TaskSpec(model="big")},
    )
    warnings = validate_config(c)
    assert any("footprint" in w for w in warnings)


def test_step_model_that_can_never_fit_fails():
    bad = cfg(
        vram_budget_mb=2000,
        models={
            "fast": ModelSpec(ref="qwen2.5:1.5b", footprint_mb=900),
            "big": ModelSpec(ref="llama3.1:8b", footprint_mb=5000),
        },
        tasks={
            "pipeline": TaskSpec(
                steps=[StepSpec(model="fast"), StepSpec(model="big")]
            )
        },
    )
    with pytest.raises(ConfigError, match="can never fit"):
        validate_config(bad)


def test_step_pipeline_that_fits_passes():
    ok = cfg(
        vram_budget_mb=6800,
        models={
            "fast": ModelSpec(ref="qwen2.5:1.5b", residency="pinned", footprint_mb=900),
            "big": ModelSpec(ref="llama3.1:8b", footprint_mb=5000),
        },
        tasks={
            "pipeline": TaskSpec(
                steps=[StepSpec(model="fast"), StepSpec(model="big")]
            )
        },
    )
    assert validate_config(ok) == []
