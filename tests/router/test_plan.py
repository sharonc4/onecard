import pytest

from onecard.config.schema import Config, ModelSpec, StepSpec, TaskSpec
from onecard.errors import ConfigError
from onecard.router.plan import build_plan


def cfg() -> Config:
    return Config(
        vram_budget_mb=8000,
        models={
            "fast": ModelSpec(ref="qwen2.5:1.5b", residency="pinned", footprint_mb=900),
            "coder": ModelSpec(ref="qwen2.5-coder:7b", footprint_mb=4500),
        },
        tasks={
            "chat": TaskSpec(model="coder", params={"num_ctx": 4096}),
            "review": TaskSpec(
                steps=[
                    StepSpec(model="coder", prompt="find"),
                    StepSpec(model="coder", prompt="rank"),
                    StepSpec(model="fast", prompt="format"),
                ]
            ),
            "picture": TaskSpec(workflow="workflows/sd15.json", exclusive=True),
        },
    )


def test_single_model_task_produces_one_step():
    plan = build_plan(cfg(), "chat")
    assert len(plan.steps) == 1
    assert plan.steps[0].model_ref == "qwen2.5-coder:7b"
    assert plan.exclusive is False


def test_need_mb_includes_kv_for_declared_context():
    # 4500 footprint + ceil(4096/1024)*64 = 4500 + 256
    assert build_plan(cfg(), "chat").steps[0].need_mb == 4756


def test_pipeline_preserves_step_order():
    assert [s.prompt for s in build_plan(cfg(), "review").steps] == ["find", "rank", "format"]


def test_consecutive_same_model_steps_collapse_into_one_load():
    assert build_plan(cfg(), "review").load_sequence == ["coder", "fast"]


def test_workflow_task_is_exclusive_and_has_no_model_steps():
    plan = build_plan(cfg(), "picture")
    assert plan.exclusive is True
    assert plan.workflow == "workflows/sd15.json"
    assert plan.steps == []


def test_unknown_task_raises():
    with pytest.raises(ConfigError, match="unknown task 'ghost'"):
        build_plan(cfg(), "ghost")


def test_step_params_override_task_params():
    c = cfg()
    c.tasks["review"].params = {"num_ctx": 2048, "temperature": 0.9}
    c.tasks["review"].steps[0].params = {"temperature": 0.1}
    plan = build_plan(c, "review")
    assert plan.steps[0].params["temperature"] == 0.1
    assert plan.steps[0].params["num_ctx"] == 2048
    assert plan.steps[1].params["temperature"] == 0.9
