from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Residency = Literal["pinned", "on_demand"]


class Strict(BaseModel):
    """Base: reject unknown keys so typos fail loudly instead of being ignored."""

    model_config = ConfigDict(extra="forbid")


class ModelSpec(Strict):
    ref: str
    residency: Residency = "on_demand"
    footprint_mb: int | None = None


class MemorySpec(Strict):
    read: bool = True
    write: bool = False


class StepSpec(Strict):
    model: str
    prompt: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Defaults(Strict):
    model: str | None = None
    memory: MemorySpec = Field(default_factory=MemorySpec)
    permissions: list[str] = Field(default_factory=list)


class TaskSpec(Strict):
    model: str | None = None
    steps: list[StepSpec] | None = None
    workflow: str | None = None
    exclusive: bool = False
    prompt: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    memory: MemorySpec = Field(default_factory=MemorySpec)


class Config(Strict):
    vram_budget_mb: int
    disk_budget_gb: int | None = None
    kv_mb_per_1k_ctx: int = 64
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    tasks: dict[str, TaskSpec] = Field(default_factory=dict)
    defaults: Defaults = Field(default_factory=Defaults)
