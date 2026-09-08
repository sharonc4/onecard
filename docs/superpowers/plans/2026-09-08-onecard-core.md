# onecard Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working `onecard` CLI that validates a YAML config, routes a named task to a specific Ollama model, arbitrates GPU residency within a declared VRAM budget, and streams the response.

**Architecture:** A pure-logic core (config, arbiter, router) that knows nothing about HTTP, tested entirely against a fake GPU consumer. One real consumer implementation (Ollama) behind a `GpuConsumer` protocol. A thin Typer CLI on top. Everything async.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx, Typer, PyYAML, pytest + pytest-asyncio, ruff, mypy. SQLite via stdlib.

**Spec:** `docs/superpowers/specs/2026-09-08-onecard-design.md`

## Global Constraints

- Python **3.11 or newer**.
- The `onecard` package **must never import torch, CUDA bindings, or any GPU library.** It talks to backends over HTTP only. A test enforces this.
- **No silent degradation anywhere.** Never substitute a different model, never swallow a backend error, never return an empty result where an error belongs. Every failure path raises a typed exception carrying the real cause.
- **Eviction is confirmed, never assumed.** After requesting a release, re-query the consumer and verify. This is the single most important invariant in the system.
- All VRAM figures are in **MiB** (`_mb` suffix), all disk figures in **GiB** (`_gb`).
- Public functions carry type annotations; `mypy --strict` passes on `src/`.
- Line length 100, formatted and linted by `ruff`.
- Commit after every task. Conventional-commit prefixes (`feat:`, `test:`, `chore:`, `docs:`).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Packaging, deps, tool config |
| `src/onecard/errors.py` | Typed exception hierarchy |
| `src/onecard/config/schema.py` | Pydantic models for `onecard.yaml` |
| `src/onecard/config/loader.py` | Read YAML → `Config`, raise on parse errors |
| `src/onecard/config/validate.py` | Cross-field rules (refs resolve, budget fits) |
| `src/onecard/gpu/consumer.py` | `Residency`, `GpuConsumer` protocol |
| `src/onecard/gpu/fake.py` | `FakeConsumer` for tests (can lie about eviction) |
| `src/onecard/gpu/arbiter.py` | Budget accounting, LRU eviction, exclusive claims |
| `src/onecard/gpu/estimate.py` | KV-cache size estimation |
| `src/onecard/router/plan.py` | Task → ordered `ExecStep` list, swap grouping |
| `src/onecard/router/execute.py` | Runs a plan through the arbiter |
| `src/onecard/backend/ollama.py` | Ollama `GpuConsumer` + chat streaming |
| `src/onecard/store/footprints.py` | SQLite measured-footprint cache |
| `src/onecard/cli/main.py` | Typer app: `validate`, `ps`, `run` |
| `tests/` | Mirrors `src/onecard/` |

---

### Task 1: Project scaffold, tooling, and CI

**Files:**
- Create: `pyproject.toml`, `src/onecard/__init__.py`, `src/onecard/errors.py`, `LICENSE`, `.gitignore`, `.github/workflows/ci.yml`
- Test: `tests/test_import_hygiene.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `onecard.errors.OneCardError`, `ConfigError`, `BudgetError`, `BackendError`, `EvictionError` — all subclasses of `OneCardError`. Every later task raises from this hierarchy.

- [ ] **Step 1: Write the failing test**

`tests/test_import_hygiene.py`:

```python
import subprocess
import sys


def test_package_imports_no_gpu_libraries():
    """The onecard package must never pull in torch/CUDA. It talks HTTP only."""
    code = (
        "import onecard, sys;"
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        "{'torch','tensorflow','pycuda','cupy','nvidia'}];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"GPU libraries imported: {out.stdout}"


def test_error_hierarchy():
    from onecard.errors import (
        BackendError,
        BudgetError,
        ConfigError,
        EvictionError,
        OneCardError,
    )

    for cls in (ConfigError, BudgetError, BackendError, EvictionError):
        assert issubclass(cls, OneCardError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_import_hygiene.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard'`

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "onecard"
version = "0.1.0"
description = "A local AI assistant for one small GPU."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
dependencies = [
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "httpx>=0.27",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "ruff>=0.5", "mypy>=1.10"]

[project.scripts]
onecard = "onecard.cli.main:app"

[tool.hatch.build.targets.wheel]
packages = ["src/onecard"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src"]

[tool.mypy]
strict = true
files = ["src"]
```

`src/onecard/__init__.py`:

```python
__version__ = "0.1.0"
```

`src/onecard/errors.py`:

```python
class OneCardError(Exception):
    """Base for every error onecard raises deliberately."""


class ConfigError(OneCardError):
    """The configuration is invalid. Raised at validate time, never mid-request."""


class BudgetError(OneCardError):
    """A claim cannot fit within the declared VRAM budget."""


class BackendError(OneCardError):
    """A backend was unreachable or returned an error."""


class EvictionError(OneCardError):
    """A consumer failed to release the GPU when asked."""
```

`.gitignore`:

```
__pycache__/
*.py[cod]
.venv/
venv/
dist/
build/
*.egg-info/
.mypy_cache/
.ruff_cache/
.pytest_cache/
data/
```

`LICENSE`: the standard Apache License 2.0 text, copyright line `Copyright 2026 onecard contributors`. Fetch verbatim from https://www.apache.org/licenses/LICENSE-2.0.txt — do not paraphrase or abbreviate it.

`.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: mypy
      - run: pytest -v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/test_import_hygiene.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src LICENSE .gitignore .github tests
git commit -m "chore: scaffold package, tooling, CI, and error hierarchy"
```

---

### Task 2: Config schema

**Files:**
- Create: `src/onecard/config/__init__.py`, `src/onecard/config/schema.py`, `src/onecard/config/loader.py`
- Test: `tests/config/test_schema.py`

**Interfaces:**
- Consumes: `onecard.errors.ConfigError`.
- Produces:
  - `ModelSpec(ref: str, residency: Residency, footprint_mb: int | None)`
  - `StepSpec(model: str, prompt: str | None, params: dict[str, Any])`
  - `TaskSpec(model, steps, workflow, exclusive, params, prompt, permissions, tools, memory)`
  - `Config(vram_budget_mb, disk_budget_gb, kv_mb_per_1k_ctx, models, tasks, defaults)`
  - `load_config(path: Path) -> Config`

- [ ] **Step 1: Write the failing test**

`tests/config/test_schema.py`:

```python
import textwrap
from pathlib import Path

import pytest

from onecard.config.loader import load_config
from onecard.errors import ConfigError

MINIMAL = textwrap.dedent(
    """
    vram_budget_mb: 6800
    models:
      fast:     { ref: "qwen2.5:1.5b", residency: pinned }
      reasoner: { ref: "llama3.1:8b" }
    tasks:
      chat:
        model: reasoner
      summarize:
        model: fast
        params: { num_ctx: 4096 }
    """
)


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "onecard.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_minimal_config(tmp_path: Path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.vram_budget_mb == 6800
    assert cfg.models["fast"].residency == "pinned"
    assert cfg.models["reasoner"].residency == "on_demand"  # defaulted
    assert cfg.tasks["summarize"].params["num_ctx"] == 4096


def test_defaults_are_applied(tmp_path: Path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.disk_budget_gb is None
    assert cfg.kv_mb_per_1k_ctx == 64
    assert cfg.tasks["chat"].permissions == []


def test_unknown_key_is_rejected(tmp_path: Path):
    body = MINIMAL + "\nnonsense_key: 1\n"
    with pytest.raises(ConfigError, match="nonsense_key"):
        load_config(write(tmp_path, body))


def test_malformed_yaml_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="parse"):
        load_config(write(tmp_path, "vram_budget_mb: [unclosed"))


def test_missing_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.config'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/config/__init__.py`: empty file.

`src/onecard/config/schema.py`:

```python
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
```

`src/onecard/config/loader.py`:

```python
from pathlib import Path

import yaml
from pydantic import ValidationError

from onecard.config.schema import Config
from onecard.errors import ConfigError


def load_config(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"could not parse {path}: top level must be a mapping")
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/config/test_schema.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/config tests/config
git commit -m "feat: config schema and loader"
```

---

### Task 3: KV cache estimation

> Implemented before config validation because `validate.py` imports it.

**Files:**
- Create: `src/onecard/gpu/__init__.py`, `src/onecard/gpu/estimate.py`
- Test: `tests/gpu/test_estimate.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `estimate_kv_mb(num_ctx: int, mb_per_1k: int = 64) -> int`.

- [ ] **Step 1: Write the failing test**

`tests/gpu/test_estimate.py`:

```python
import pytest

from onecard.gpu.estimate import estimate_kv_mb


def test_scales_linearly_with_context():
    assert estimate_kv_mb(1024, 64) == 64
    assert estimate_kv_mb(4096, 64) == 256


def test_partial_blocks_round_up():
    """Never under-estimate: a partial 1k block still costs a full block."""
    assert estimate_kv_mb(1500, 64) == 128


def test_zero_context_costs_nothing():
    assert estimate_kv_mb(0, 64) == 0


def test_rate_is_configurable():
    assert estimate_kv_mb(2048, 100) == 200


def test_negative_context_is_rejected():
    with pytest.raises(ValueError):
        estimate_kv_mb(-1, 64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gpu/test_estimate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.gpu'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/gpu/__init__.py`: empty file.

`src/onecard/gpu/estimate.py`:

```python
import math

DEFAULT_KV_MB_PER_1K = 64


def estimate_kv_mb(num_ctx: int, mb_per_1k: int = DEFAULT_KV_MB_PER_1K) -> int:
    """Estimate KV-cache VRAM for a context window.

    A deliberate over-estimate: partial 1k blocks round up, because
    under-estimating causes a spill to system RAM, which is the failure
    this project exists to prevent. The rate is configurable per deployment
    (`kv_mb_per_1k_ctx`) because it varies by model architecture.
    """
    if num_ctx < 0:
        raise ValueError("num_ctx must be non-negative")
    return math.ceil(num_ctx / 1024) * mb_per_1k
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gpu/test_estimate.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/gpu tests/gpu
git commit -m "feat: KV cache estimation that rounds up rather than under-committing"
```

---

### Task 4: Config validation rules

**Files:**
- Create: `src/onecard/config/validate.py`
- Test: `tests/config/test_validate.py`

**Interfaces:**
- Consumes: `Config`, `TaskSpec`, `ConfigError`, `estimate_kv_mb`.
- Produces: `validate_config(cfg: Config) -> list[str]` — returns warnings, raises `ConfigError` on anything fatal.

- [ ] **Step 1: Write the failing test**

`tests/config/test_validate.py`:

```python
import pytest

from onecard.config.schema import Config, ModelSpec, StepSpec, TaskSpec
from onecard.config.validate import validate_config
from onecard.errors import ConfigError


def cfg(**kw) -> Config:
    base = dict(
        vram_budget_mb=6800,
        models={
            "fast": ModelSpec(ref="qwen2.5:1.5b", residency="pinned", footprint_mb=900),
            "big": ModelSpec(ref="llama3.1:8b", footprint_mb=5000),
        },
        tasks={"chat": TaskSpec(model="big")},
    )
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/config/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.config.validate'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/config/validate.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/config -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/config/validate.py tests/config/test_validate.py
git commit -m "feat: config validation catches overcommit and dangling refs at validate time"
```

---

### Task 5: GPU consumer protocol and fake

**Files:**
- Create: `src/onecard/gpu/consumer.py`, `src/onecard/gpu/fake.py`
- Test: `tests/gpu/test_fake.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Residency(consumer: str, key: str, footprint_mb: int, pinned: bool = False)`
  - `DEFAULT_FOOTPRINT_MB = 4500`
  - `GpuConsumer` protocol: `name: str`, `async residents() -> list[Residency]`, `async load(key, *, pinned, footprint_hint_mb) -> Residency`, `async release(key) -> None`, `async release_all() -> None`
  - `FakeConsumer(name, footprints: dict[str, int], honest: bool = True)` with `load_count: dict[str, int]` and `release_all_count: int`

- [ ] **Step 1: Write the failing test**

`tests/gpu/test_fake.py`:

```python
from onecard.gpu.fake import FakeConsumer


async def test_load_then_residents_reports_it():
    c = FakeConsumer("fake", {"a": 100})
    r = await c.load("a", pinned=False, footprint_hint_mb=None)
    assert r.key == "a"
    assert r.footprint_mb == 100
    assert [x.key for x in await c.residents()] == ["a"]


async def test_release_removes_it():
    c = FakeConsumer("fake", {"a": 100})
    await c.load("a", pinned=False, footprint_hint_mb=None)
    await c.release("a")
    assert await c.residents() == []


async def test_release_all_clears_even_pinned():
    c = FakeConsumer("fake", {"a": 100, "b": 200})
    await c.load("a", pinned=True, footprint_hint_mb=None)
    await c.load("b", pinned=False, footprint_hint_mb=None)
    await c.release_all()
    assert await c.residents() == []


async def test_dishonest_consumer_keeps_reporting_after_release():
    """Simulates a backend that acknowledges eviction but has not actually freed VRAM."""
    c = FakeConsumer("fake", {"a": 100}, honest=False)
    await c.load("a", pinned=False, footprint_hint_mb=None)
    await c.release("a")
    assert [x.key for x in await c.residents()] == ["a"]


async def test_unknown_key_uses_hint():
    c = FakeConsumer("fake", {})
    r = await c.load("ghost", pinned=False, footprint_hint_mb=4321)
    assert r.footprint_mb == 4321


async def test_load_counts_are_recorded():
    c = FakeConsumer("fake", {"a": 100})
    await c.load("a", pinned=False, footprint_hint_mb=None)
    await c.release("a")
    await c.load("a", pinned=False, footprint_hint_mb=None)
    assert c.load_count["a"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gpu/test_fake.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.gpu.fake'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/gpu/consumer.py`:

```python
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

DEFAULT_FOOTPRINT_MB = 4500


@dataclass(frozen=True)
class Residency:
    consumer: str
    key: str
    footprint_mb: int
    pinned: bool = False


@runtime_checkable
class GpuConsumer(Protocol):
    """Anything that can hold the GPU. Ollama and ComfyUI both implement this."""

    name: str

    async def residents(self) -> list[Residency]:
        """Ground truth: what this consumer currently holds."""

    async def load(
        self, key: str, *, pinned: bool, footprint_hint_mb: int | None
    ) -> Residency: ...

    async def release(self, key: str) -> None: ...

    async def release_all(self) -> None: ...
```

`src/onecard/gpu/fake.py`:

```python
from collections import defaultdict

from onecard.gpu.consumer import DEFAULT_FOOTPRINT_MB, Residency


class FakeConsumer:
    """In-memory GpuConsumer for tests.

    honest=False models a backend that acknowledges a release but has not
    actually freed the memory — the case the arbiter must detect.
    """

    def __init__(self, name: str, footprints: dict[str, int], honest: bool = True) -> None:
        self.name = name
        self.footprints = footprints
        self.honest = honest
        self._resident: dict[str, Residency] = {}
        self.load_count: dict[str, int] = defaultdict(int)
        self.release_all_count = 0

    async def residents(self) -> list[Residency]:
        return list(self._resident.values())

    async def load(
        self, key: str, *, pinned: bool, footprint_hint_mb: int | None
    ) -> Residency:
        mb = self.footprints.get(key, footprint_hint_mb or DEFAULT_FOOTPRINT_MB)
        r = Residency(consumer=self.name, key=key, footprint_mb=mb, pinned=pinned)
        self._resident[key] = r
        self.load_count[key] += 1
        return r

    async def release(self, key: str) -> None:
        if self.honest:
            self._resident.pop(key, None)

    async def release_all(self) -> None:
        self.release_all_count += 1
        if self.honest:
            self._resident.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gpu/test_fake.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/gpu/consumer.py src/onecard/gpu/fake.py tests/gpu/test_fake.py
git commit -m "feat: GpuConsumer protocol and fake consumer that can lie about eviction"
```

---

### Task 6: Arbiter — budget accounting and LRU eviction

**Files:**
- Create: `src/onecard/gpu/arbiter.py`
- Test: `tests/gpu/test_arbiter.py`

**Interfaces:**
- Consumes: `Residency`, `GpuConsumer`, `BudgetError`, `EvictionError`.
- Produces:
  - `SwapEvent(evicted: str, reason: str, duration_s: float)`
  - `Arbiter(budget_mb: int, consumers: dict[str, GpuConsumer])`
  - `async claim(consumer, key, need_mb, *, pinned=False, exclusive=False) -> Residency`
  - `async residents() -> list[Residency]`, `async used_mb() -> int`, `swaps: list[SwapEvent]`

- [ ] **Step 1: Write the failing test**

`tests/gpu/test_arbiter.py`:

```python
import pytest

from onecard.errors import BudgetError
from onecard.gpu.arbiter import Arbiter
from onecard.gpu.fake import FakeConsumer


def arb(budget=1000, **footprints) -> tuple[Arbiter, FakeConsumer]:
    c = FakeConsumer("ollama", dict(footprints))
    return Arbiter(budget_mb=budget, consumers={"ollama": c}), c


async def test_claim_that_fits_loads_without_eviction():
    a, c = arb(budget=1000, a=400)
    await a.claim("ollama", "a", need_mb=400)
    assert [r.key for r in await a.residents()] == ["a"]
    assert a.swaps == []


async def test_second_claim_that_fits_keeps_both():
    a, c = arb(budget=1000, a=400, b=400)
    await a.claim("ollama", "a", need_mb=400)
    await a.claim("ollama", "b", need_mb=400)
    assert {r.key for r in await a.residents()} == {"a", "b"}


async def test_claim_evicts_least_recently_used():
    a, c = arb(budget=1000, a=400, b=400, d=400)
    await a.claim("ollama", "a", need_mb=400)
    await a.claim("ollama", "b", need_mb=400)
    await a.claim("ollama", "a", need_mb=400)   # touch 'a' so 'b' is now LRU
    await a.claim("ollama", "d", need_mb=400)
    assert {r.key for r in await a.residents()} == {"a", "d"}
    assert a.swaps[0].evicted == "b"


async def test_pinned_models_are_not_evicted_by_ordinary_claims():
    a, c = arb(budget=1000, p=400, x=400, y=400)
    await a.claim("ollama", "p", need_mb=400, pinned=True)
    await a.claim("ollama", "x", need_mb=400)
    await a.claim("ollama", "y", need_mb=400)
    assert "p" in {r.key for r in await a.residents()}
    assert a.swaps[0].evicted == "x"


async def test_claim_larger_than_budget_raises_and_loads_nothing():
    a, c = arb(budget=1000, huge=2000)
    with pytest.raises(BudgetError, match="exceeds the budget"):
        await a.claim("ollama", "huge", need_mb=2000)
    assert await a.residents() == []


async def test_claim_that_cannot_fit_around_pinned_raises():
    a, c = arb(budget=1000, p=800, big=400)
    await a.claim("ollama", "p", need_mb=800, pinned=True)
    with pytest.raises(BudgetError, match="cannot fit"):
        await a.claim("ollama", "big", need_mb=400)


async def test_reclaiming_a_resident_model_does_not_reload_it():
    a, c = arb(budget=1000, a=400)
    await a.claim("ollama", "a", need_mb=400)
    await a.claim("ollama", "a", need_mb=400)
    assert c.load_count["a"] == 1


async def test_unknown_consumer_raises():
    a, _ = arb()
    with pytest.raises(KeyError):
        await a.claim("nope", "a", need_mb=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gpu/test_arbiter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.gpu.arbiter'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/gpu/arbiter.py`:

```python
import asyncio
import time
from dataclasses import dataclass, field

from onecard.errors import BudgetError, EvictionError
from onecard.gpu.consumer import GpuConsumer, Residency


@dataclass
class SwapEvent:
    evicted: str
    reason: str
    duration_s: float


@dataclass
class _Held:
    residency: Residency
    last_used: float = field(default_factory=time.monotonic)


class Arbiter:
    """Owns the GPU. Decides who holds it and enforces the VRAM budget.

    Every consumer of the card goes through here. The arbiter reasons about
    claims, never about model families.
    """

    def __init__(self, budget_mb: int, consumers: dict[str, GpuConsumer]) -> None:
        self.budget_mb = budget_mb
        self.consumers = consumers
        self.swaps: list[SwapEvent] = []
        self._held: dict[tuple[str, str], _Held] = {}
        self._lock = asyncio.Lock()

    async def residents(self) -> list[Residency]:
        return [h.residency for h in self._held.values()]

    async def used_mb(self) -> int:
        return sum(h.residency.footprint_mb for h in self._held.values())

    async def claim(
        self,
        consumer: str,
        key: str,
        need_mb: int,
        *,
        pinned: bool = False,
        exclusive: bool = False,
    ) -> Residency:
        if consumer not in self.consumers:
            raise KeyError(f"unknown GPU consumer: {consumer}")
        async with self._lock:
            return await self._claim_locked(
                consumer, key, need_mb, pinned=pinned, exclusive=exclusive
            )

    async def _claim_locked(
        self, consumer: str, key: str, need_mb: int, *, pinned: bool, exclusive: bool
    ) -> Residency:
        if need_mb > self.budget_mb:
            raise BudgetError(
                f"'{key}' needs {need_mb}MB which exceeds the budget of {self.budget_mb}MB"
            )

        ident = (consumer, key)
        if not exclusive and ident in self._held:
            self._held[ident].last_used = time.monotonic()
            return self._held[ident].residency

        if exclusive:
            await self._release_everything(reason=f"exclusive claim by '{key}'")

        while await self.used_mb() + need_mb > self.budget_mb:
            victim = self._lru_victim()
            if victim is None:
                raise BudgetError(
                    f"'{key}' needs {need_mb}MB and cannot fit: "
                    f"{await self.used_mb()}MB is held by models that cannot be evicted"
                )
            await self._evict(victim, reason=f"making room for '{key}'")

        residency = await self.consumers[consumer].load(
            key, pinned=pinned, footprint_hint_mb=need_mb
        )
        self._held[ident] = _Held(residency=residency)
        return residency

    def _lru_victim(self) -> tuple[str, str] | None:
        candidates = [k for k, h in self._held.items() if not h.residency.pinned]
        if not candidates:
            return None
        return min(candidates, key=lambda k: self._held[k].last_used)

    async def _evict(self, ident: tuple[str, str], *, reason: str) -> None:
        consumer, key = ident
        started = time.monotonic()
        await self.consumers[consumer].release(key)
        await self._confirm_gone(consumer, key)
        self._held.pop(ident, None)
        self.swaps.append(
            SwapEvent(evicted=key, reason=reason, duration_s=time.monotonic() - started)
        )

    async def _confirm_gone(self, consumer: str, key: str) -> None:
        """Confirm the release actually happened. Never assume it did."""
        still = [r.key for r in await self.consumers[consumer].residents()]
        if key in still:
            raise EvictionError(
                f"consumer '{consumer}' still reports '{key}' as resident after release; "
                "refusing to proceed rather than risk spilling to system RAM"
            )

    async def _release_everything(self, *, reason: str) -> None:
        for name, consumer in self.consumers.items():
            started = time.monotonic()
            await consumer.release_all()
            remaining = await consumer.residents()
            if remaining:
                raise EvictionError(
                    f"consumer '{name}' still holds "
                    f"{[r.key for r in remaining]} after release_all; "
                    "refusing to grant an exclusive claim"
                )
            self.swaps.append(
                SwapEvent(
                    evicted=f"{name}:*",
                    reason=reason,
                    duration_s=time.monotonic() - started,
                )
            )
        self._held.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gpu/test_arbiter.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/gpu/arbiter.py tests/gpu/test_arbiter.py
git commit -m "feat: arbiter with budget accounting, LRU eviction, and confirmed releases"
```

---

### Task 7: Arbiter — exclusive claims and pinned restore

**Files:**
- Modify: `src/onecard/gpu/arbiter.py` (add `restore_pinned`)
- Test: `tests/gpu/test_arbiter_exclusive.py`

**Interfaces:**
- Consumes: everything from Task 6.
- Produces: `async restore_pinned(specs: list[tuple[str, str, int]]) -> None` where each tuple is `(consumer, key, need_mb)`.

- [ ] **Step 1: Write the failing test**

`tests/gpu/test_arbiter_exclusive.py`:

```python
import asyncio

import pytest

from onecard.errors import EvictionError
from onecard.gpu.arbiter import Arbiter
from onecard.gpu.fake import FakeConsumer


def two_consumer_arb(budget=8000):
    ollama = FakeConsumer("ollama", {"small": 900, "big": 5000})
    comfy = FakeConsumer("comfyui", {"sd15": 4000, "sd15b": 4000})
    arb = Arbiter(budget_mb=budget, consumers={"ollama": ollama, "comfyui": comfy})
    return arb, ollama, comfy


async def test_exclusive_claim_evicts_everything_including_pinned():
    a, ollama, comfy = two_consumer_arb()
    await a.claim("ollama", "small", need_mb=900, pinned=True)
    await a.claim("ollama", "big", need_mb=5000)
    await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)

    assert [r.key for r in await a.residents()] == ["sd15"]
    assert await ollama.residents() == []


async def test_exclusive_claim_fails_loudly_if_a_consumer_will_not_release():
    liar = FakeConsumer("ollama", {"stuck": 5000}, honest=False)
    comfy = FakeConsumer("comfyui", {"sd15": 4000})
    a = Arbiter(budget_mb=8000, consumers={"ollama": liar, "comfyui": comfy})
    await a.claim("ollama", "stuck", need_mb=5000)

    with pytest.raises(EvictionError, match="still holds"):
        await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)

    assert await comfy.residents() == [], "must not load while the card is still occupied"


async def test_restore_pinned_reloads_after_exclusive_work():
    a, ollama, comfy = two_consumer_arb()
    await a.claim("ollama", "small", need_mb=900, pinned=True)
    await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)

    await a.restore_pinned([("ollama", "small", 900)])
    assert "small" in {r.key for r in await a.residents()}


async def test_restore_pinned_is_idempotent():
    a, ollama, comfy = two_consumer_arb()
    await a.claim("ollama", "small", need_mb=900, pinned=True)
    await a.restore_pinned([("ollama", "small", 900)])
    assert ollama.load_count["small"] == 1


async def test_concurrent_exclusive_claims_serialize():
    a, ollama, comfy = two_consumer_arb()
    order: list[str] = []

    async def claim(key: str) -> None:
        await a.claim("comfyui", key, need_mb=4000, exclusive=True)
        order.append(f"done:{key}")

    await asyncio.gather(claim("sd15"), claim("sd15b"))
    assert len(order) == 2, "both claims must complete, one after the other"
    assert len([r for r in await a.residents()]) == 1, "only one may hold the card"


async def test_swap_events_record_exclusive_reason():
    a, ollama, comfy = two_consumer_arb()
    await a.claim("ollama", "big", need_mb=5000)
    await a.claim("comfyui", "sd15", need_mb=4000, exclusive=True)
    assert any("exclusive claim" in s.reason for s in a.swaps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gpu/test_arbiter_exclusive.py -v`
Expected: FAIL — `test_restore_pinned_reloads_after_exclusive_work` errors with `AttributeError: 'Arbiter' object has no attribute 'restore_pinned'`

- [ ] **Step 3: Write minimal implementation**

Append this method to `class Arbiter` in `src/onecard/gpu/arbiter.py`:

```python
    async def restore_pinned(self, specs: list[tuple[str, str, int]]) -> None:
        """Reload pinned models after an exclusive claim released them.

        Failures here are raised, not swallowed: a harness that silently came
        back without its pinned fast model would make every later request
        mysteriously slower with no explanation.
        """
        async with self._lock:
            for consumer, key, need_mb in specs:
                if (consumer, key) in self._held:
                    continue
                await self._claim_locked(
                    consumer, key, need_mb, pinned=True, exclusive=False
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gpu -v`
Expected: 25 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/gpu/arbiter.py tests/gpu/test_arbiter_exclusive.py
git commit -m "feat: exclusive claims evict everything, serialize, and restore pinned models"
```

---

### Task 8: Router — task to execution plan with swap grouping

**Files:**
- Create: `src/onecard/router/__init__.py`, `src/onecard/router/plan.py`
- Test: `tests/router/test_plan.py`

**Interfaces:**
- Consumes: `Config`, `TaskSpec`, `estimate_kv_mb`, `DEFAULT_FOOTPRINT_MB`, `ConfigError`.
- Produces:
  - `ExecStep(model_name: str, model_ref: str, prompt: str | None, params: dict[str, Any], need_mb: int)`
  - `ExecPlan(task: str, steps: list[ExecStep], exclusive: bool, workflow: str | None)`
  - `ExecPlan.load_sequence -> list[str]`
  - `build_plan(cfg: Config, task_name: str) -> ExecPlan`

- [ ] **Step 1: Write the failing test**

`tests/router/test_plan.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/router/test_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.router'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/router/__init__.py`: empty file.

`src/onecard/router/plan.py`:

```python
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
    footprint = spec.footprint_mb if spec.footprint_mb is not None else DEFAULT_FOOTPRINT_MB
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/router/test_plan.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/router tests/router
git commit -m "feat: router builds execution plans that collapse consecutive same-model steps"
```

---

### Task 9: Ollama consumer

**Files:**
- Create: `src/onecard/backend/__init__.py`, `src/onecard/backend/ollama.py`
- Test: `tests/backend/test_ollama.py`

**Interfaces:**
- Consumes: `Residency`, `DEFAULT_FOOTPRINT_MB`, `BackendError`.
- Produces:
  - `OllamaConsumer(base_url: str, client: httpx.AsyncClient)` implementing `GpuConsumer`, `name = "ollama"`
  - `async chat(model_ref: str, prompt: str, params: dict[str, Any]) -> AsyncIterator[str]`

Ollama endpoints used: `GET /api/ps`, `POST /api/generate` with `keep_alive` (`-1` pins, `0` evicts), `POST /api/chat` with `stream: true`.

- [ ] **Step 1: Write the failing test**

`tests/backend/test_ollama.py`:

```python
import json

import httpx
import pytest

from onecard.backend.ollama import OllamaConsumer
from onecard.errors import BackendError

PS_TWO = {
    "models": [
        {"name": "llama3.1:8b", "size_vram": 5_000_000_000},
        {"name": "qwen2.5:1.5b", "size_vram": 900_000_000},
    ]
}


def consumer(handler) -> OllamaConsumer:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama:11434"
    )
    return OllamaConsumer(base_url="http://ollama:11434", client=client)


async def test_residents_reports_vram_in_mib():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json=PS_TWO)

    residents = await consumer(handler).residents()
    assert {r.key for r in residents} == {"llama3.1:8b", "qwen2.5:1.5b"}
    big = next(r for r in residents if r.key == "llama3.1:8b")
    assert big.footprint_mb == 4768  # 5e9 bytes // 1024^2
    assert big.consumer == "ollama"


async def test_load_sends_keep_alive_minus_one_when_pinned():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"done": True})
        return httpx.Response(200, json=PS_TWO)

    await consumer(handler).load("llama3.1:8b", pinned=True, footprint_hint_mb=None)
    assert seen["keep_alive"] == -1
    assert seen["model"] == "llama3.1:8b"


async def test_release_sends_keep_alive_zero():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            seen.update(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    await consumer(handler).release("llama3.1:8b")
    assert seen["keep_alive"] == 0


async def test_release_all_releases_every_resident_model():
    released: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=PS_TWO)
        released.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"done": True})

    await consumer(handler).release_all()
    assert sorted(released) == ["llama3.1:8b", "qwen2.5:1.5b"]


async def test_unreachable_backend_raises_backend_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(BackendError, match="unreachable"):
        await consumer(handler).residents()


async def test_http_error_raises_backend_error_with_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(BackendError, match="500"):
        await consumer(handler).residents()


async def test_load_that_does_not_become_resident_raises():
    """Never report success for a load the backend did not actually perform."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(200, json={"models": []})

    with pytest.raises(BackendError, match="does not report it as resident"):
        await consumer(handler).load("llama3.1:8b", pinned=False, footprint_hint_mb=None)


async def test_chat_streams_content_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            '{"message":{"content":"Hel"},"done":false}\n'
            '{"message":{"content":"lo"},"done":false}\n'
            '{"message":{"content":""},"done":true}\n'
        )
        return httpx.Response(200, text=body)

    chunks = [c async for c in consumer(handler).chat("m", "hi", {})]
    assert "".join(chunks) == "Hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_ollama.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.backend'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/backend/__init__.py`: empty file.

`src/onecard/backend/ollama.py`:

```python
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from onecard.errors import BackendError
from onecard.gpu.consumer import DEFAULT_FOOTPRINT_MB, Residency

BYTES_PER_MIB = 1024 * 1024


class OllamaConsumer:
    """GpuConsumer backed by an Ollama server.

    Residency policy is expressed entirely through `keep_alive`: -1 pins,
    0 evicts immediately, a duration sets a timeout. Ground truth about what
    is loaded always comes from /api/ps, never from our own bookkeeping.
    """

    name = "ollama"

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            resp = await self._client.post(path, json=payload, timeout=300.0)
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama unreachable at {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        return resp

    async def residents(self) -> list[Residency]:
        try:
            resp = await self._client.get("/api/ps", timeout=30.0)
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama unreachable at {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        return [
            Residency(
                consumer=self.name,
                key=entry["name"],
                footprint_mb=int(entry.get("size_vram", 0)) // BYTES_PER_MIB,
            )
            for entry in resp.json().get("models", [])
        ]

    async def load(
        self, key: str, *, pinned: bool, footprint_hint_mb: int | None
    ) -> Residency:
        await self._post("/api/generate", {"model": key, "keep_alive": -1 if pinned else "10m"})
        for r in await self.residents():
            if r.key == key:
                # A measured footprint always beats an estimate we were handed.
                return Residency(
                    consumer=self.name,
                    key=key,
                    footprint_mb=r.footprint_mb
                    or (footprint_hint_mb or DEFAULT_FOOTPRINT_MB),
                    pinned=pinned,
                )
        raise BackendError(
            f"ollama accepted a load of '{key}' but does not report it as resident"
        )

    async def release(self, key: str) -> None:
        await self._post("/api/generate", {"model": key, "keep_alive": 0})

    async def release_all(self) -> None:
        for r in await self.residents():
            await self.release(r.key)

    async def chat(
        self, model_ref: str, prompt: str, params: dict[str, Any]
    ) -> AsyncIterator[str]:
        payload = {
            "model": model_ref,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": params,
        }
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=payload, timeout=600.0
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode()[:200]
                    raise BackendError(f"ollama returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    content = json.loads(line).get("message", {}).get("content", "")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama unreachable at {self.base_url}: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/backend/test_ollama.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/backend tests/backend
git commit -m "feat: Ollama consumer expressing residency through keep_alive"
```

---

### Task 10: Measured footprint store

**Files:**
- Create: `src/onecard/store/__init__.py`, `src/onecard/store/footprints.py`
- Test: `tests/store/test_footprints.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FootprintStore(db_path: Path)` with `get(consumer, key) -> int | None`, `record(consumer, key, footprint_mb) -> None`, `all() -> dict[tuple[str, str], int]`.

- [ ] **Step 1: Write the failing test**

`tests/store/test_footprints.py`:

```python
from pathlib import Path

from onecard.store.footprints import FootprintStore


def test_unknown_model_returns_none(tmp_path: Path):
    assert FootprintStore(tmp_path / "f.db").get("ollama", "ghost") is None


def test_recorded_value_is_returned(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "llama3.1:8b", 4768)
    assert s.get("ollama", "llama3.1:8b") == 4768


def test_recording_again_overwrites(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "m", 100)
    s.record("ollama", "m", 200)
    assert s.get("ollama", "m") == 200


def test_values_persist_across_instances(tmp_path: Path):
    db = tmp_path / "f.db"
    FootprintStore(db).record("ollama", "m", 321)
    assert FootprintStore(db).get("ollama", "m") == 321


def test_same_key_under_different_consumers_is_distinct(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "x", 100)
    s.record("comfyui", "x", 4000)
    assert s.get("ollama", "x") == 100
    assert s.get("comfyui", "x") == 4000


def test_all_returns_every_row(tmp_path: Path):
    s = FootprintStore(tmp_path / "f.db")
    s.record("ollama", "a", 1)
    s.record("comfyui", "b", 2)
    assert s.all() == {("ollama", "a"): 1, ("comfyui", "b"): 2}


def test_parent_directory_is_created(tmp_path: Path):
    s = FootprintStore(tmp_path / "nested" / "deeper" / "f.db")
    s.record("ollama", "m", 5)
    assert s.get("ollama", "m") == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/store/test_footprints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.store'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/store/__init__.py`: empty file.

`src/onecard/store/footprints.py`:

```python
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS footprints (
    consumer     TEXT NOT NULL,
    key          TEXT NOT NULL,
    footprint_mb INTEGER NOT NULL,
    PRIMARY KEY (consumer, key)
);
"""


class FootprintStore:
    """Measured VRAM footprints, so estimates self-correct after first load.

    Plain SQLite rows a human can read and delete — there is no hand-maintained
    size table anywhere in the codebase to go stale.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def get(self, consumer: str, key: str) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT footprint_mb FROM footprints WHERE consumer = ? AND key = ?",
                (consumer, key),
            ).fetchone()
        return int(row[0]) if row else None

    def record(self, consumer: str, key: str, footprint_mb: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO footprints (consumer, key, footprint_mb) VALUES (?, ?, ?) "
                "ON CONFLICT(consumer, key) DO UPDATE SET footprint_mb = excluded.footprint_mb",
                (consumer, key, footprint_mb),
            )

    def all(self) -> dict[tuple[str, str], int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT consumer, key, footprint_mb FROM footprints").fetchall()
        return {(r[0], r[1]): int(r[2]) for r in rows}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/store/test_footprints.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/store tests/store
git commit -m "feat: SQLite footprint store so VRAM estimates self-correct"
```

---

### Task 11: Executor — run a plan through the arbiter

**Files:**
- Create: `src/onecard/router/execute.py`
- Test: `tests/router/test_execute.py`

**Interfaces:**
- Consumes: `ExecPlan`, `Arbiter`, `FootprintStore`.
- Produces: `Executor(arbiter, chat_fn, store, on_event)` with `async run(plan: ExecPlan, user_input: str) -> AsyncIterator[str]`. `on_event(text: str)` receives arbitration progress lines.

- [ ] **Step 1: Write the failing test**

`tests/router/test_execute.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/router/test_execute.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.router.execute'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/router/execute.py`:

```python
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
```

> Note on event ordering: the "loading" line is emitted before the claim so the
> user sees *why* the pause is happening while it happens; eviction lines follow
> because the arbiter only records them once the release is confirmed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/router -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/onecard/router/execute.py tests/router/test_execute.py
git commit -m "feat: executor claims the GPU per step and chains pipeline output"
```

---

### Task 12: CLI

**Files:**
- Create: `src/onecard/cli/__init__.py`, `src/onecard/cli/main.py`
- Test: `tests/cli/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: Typer app with `validate`, `ps`, `run`. Exit code 0 on success, 1 on any `OneCardError`.

- [ ] **Step 1: Write the failing test**

`tests/cli/test_cli.py`:

```python
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from onecard.cli.main import app

runner = CliRunner()

GOOD = textwrap.dedent(
    """
    vram_budget_mb: 8000
    models:
      coder: { ref: "qwen2.5-coder:7b", footprint_mb: 4500 }
    tasks:
      code: { model: coder }
    """
)

TOO_BIG = textwrap.dedent(
    """
    vram_budget_mb: 1000
    models:
      coder: { ref: "qwen2.5-coder:7b", footprint_mb: 4500 }
    tasks:
      code: { model: coder }
    """
)


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "onecard.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_validate_accepts_a_good_config(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--config", str(write(tmp_path, GOOD))])
    assert result.exit_code == 0
    assert "ok" in result.stdout.lower()


def test_validate_rejects_an_overcommitted_config(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--config", str(write(tmp_path, TOO_BIG))])
    assert result.exit_code == 1
    assert "can never fit" in result.stdout


def test_validate_reports_missing_config_clearly(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_run_with_unknown_task_fails_clearly(tmp_path: Path):
    result = runner.invoke(
        app, ["run", "ghost", "hi", "--config", str(write(tmp_path, GOOD))]
    )
    assert result.exit_code == 1
    assert "unknown task 'ghost'" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cli/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onecard.cli'`

- [ ] **Step 3: Write minimal implementation**

`src/onecard/cli/__init__.py`: empty file.

`src/onecard/cli/main.py`:

```python
import asyncio
from pathlib import Path

import httpx
import typer

from onecard.backend.ollama import OllamaConsumer
from onecard.config.loader import load_config
from onecard.config.validate import validate_config
from onecard.errors import OneCardError
from onecard.gpu.arbiter import Arbiter
from onecard.router.execute import Executor
from onecard.router.plan import build_plan
from onecard.store.footprints import FootprintStore

app = typer.Typer(help="A local AI assistant for one small GPU.", no_args_is_help=True)

ConfigOpt = typer.Option(Path("onecard.yaml"), "--config", "-c", help="Path to onecard.yaml")
OllamaOpt = typer.Option("http://localhost:11434", "--ollama", help="Ollama base URL")
DataOpt = typer.Option(Path("data"), "--data", help="Directory for the SQLite stores")


def _fail(exc: Exception) -> None:
    typer.echo(str(exc))
    raise typer.Exit(code=1)


@app.command()
def validate(config: Path = ConfigOpt) -> None:
    """Check the config before anything tries to use it."""
    try:
        cfg = load_config(config)
        warnings = validate_config(cfg)
    except OneCardError as exc:
        _fail(exc)
        return
    for w in warnings:
        typer.echo(f"warning: {w}")
    typer.echo(
        f"ok: {len(cfg.tasks)} tasks, {len(cfg.models)} models, "
        f"budget {cfg.vram_budget_mb}MB"
    )


@app.command()
def ps(config: Path = ConfigOpt, ollama: str = OllamaOpt) -> None:
    """Show what is currently resident on the card, and what budget remains."""

    async def _run() -> None:
        cfg = load_config(config)
        async with httpx.AsyncClient(base_url=ollama) as client:
            residents = await OllamaConsumer(base_url=ollama, client=client).residents()
        used = sum(r.footprint_mb for r in residents)
        for r in residents:
            typer.echo(f"{r.consumer:8} {r.key:40} {r.footprint_mb:6}MB")
        typer.echo(f"{'':8} {'TOTAL':40} {used:6}MB of {cfg.vram_budget_mb}MB")

    try:
        asyncio.run(_run())
    except OneCardError as exc:
        _fail(exc)


@app.command()
def run(
    task: str = typer.Argument(..., help="Task name from the config"),
    text: str = typer.Argument(..., help="Input for the task"),
    config: Path = ConfigOpt,
    ollama: str = OllamaOpt,
    data: Path = DataOpt,
    explain: bool = typer.Option(False, "--explain", help="Show routing decisions"),
) -> None:
    """Run a named task."""

    async def _run() -> None:
        cfg = load_config(config)
        validate_config(cfg)
        plan = build_plan(cfg, task)
        if explain:
            typer.echo(f"task={plan.task} models={plan.load_sequence}", err=True)
        async with httpx.AsyncClient(base_url=ollama) as client:
            consumer = OllamaConsumer(base_url=ollama, client=client)
            arbiter = Arbiter(budget_mb=cfg.vram_budget_mb, consumers={"ollama": consumer})
            executor = Executor(
                arbiter=arbiter,
                chat_fn=consumer.chat,
                store=FootprintStore(data / "footprints.db"),
                on_event=lambda msg: typer.echo(msg, err=True),
            )
            async for chunk in executor.run(plan, text):
                typer.echo(chunk, nl=False)
        typer.echo("")

    try:
        asyncio.run(_run())
    except OneCardError as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -v`
Expected: whole suite green (4 CLI tests plus everything before)

- [ ] **Step 5: Commit**

```bash
git add src/onecard/cli tests/cli
git commit -m "feat: CLI with validate, ps, and streaming run"
```

---

### Task 13: Packaging, shipped configs, and README

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `docker-compose.cpu.yml`, `README.md`, `onecard.yaml`, `profiles/8gb-developer.yaml`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: `load_config`, `validate_config`.
- Produces: shipped configs proven valid by test.

- [ ] **Step 1: Write the failing test**

`tests/test_packaging.py`:

```python
from pathlib import Path

import pytest

from onecard.config.loader import load_config
from onecard.config.validate import validate_config

ROOT = Path(__file__).resolve().parents[1]


def shipped_configs() -> list[Path]:
    return [ROOT / "onecard.yaml", *sorted((ROOT / "profiles").glob("*.yaml"))]


@pytest.mark.parametrize("path", shipped_configs(), ids=lambda p: p.name)
def test_shipped_config_is_valid(path: Path):
    """Every config we ship must pass our own validator."""
    validate_config(load_config(path))


def test_at_least_one_profile_ships():
    assert len(list((ROOT / "profiles").glob("*.yaml"))) >= 1


def test_compose_declares_a_gpu_reservation():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "nvidia" in text
    assert "ollama" in text


def test_cpu_override_removes_the_reservation():
    text = (ROOT / "docker-compose.cpu.yml").read_text(encoding="utf-8")
    assert "devices: []" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_packaging.py -v`
Expected: FAIL — collection error, `profiles/` does not exist

- [ ] **Step 3: Write minimal implementation**

`onecard.yaml` (shipped default, conservative for 8GB):

```yaml
vram_budget_mb: 6800
disk_budget_gb: 60

models:
  fast:     { ref: "qwen2.5:1.5b-instruct-q4_K_M", residency: pinned, footprint_mb: 1100 }
  reasoner: { ref: "llama3.1:8b-instruct-q4_K_M", footprint_mb: 5000 }

tasks:
  chat:
    model: reasoner
    params: { num_ctx: 4096 }
  summarize:
    model: fast
    params: { num_ctx: 8192, temperature: 0.2 }
```

`profiles/8gb-developer.yaml`:

```yaml
# Profile: 8gb-developer
# Reviewed: 2026-09-08
# Evidence: sizes taken from published Q4_K_M quantizations. NOT yet measured on
#           real 8GB hardware — see the design doc's open items. If you run this
#           on a real card, please open a PR correcting these numbers.
vram_budget_mb: 6800
disk_budget_gb: 60

models:
  fast:  { ref: "qwen2.5:1.5b-instruct-q4_K_M", residency: pinned, footprint_mb: 1100 }
  coder: { ref: "qwen2.5-coder:7b-instruct-q4_K_M", footprint_mb: 4700 }

tasks:
  code:
    model: coder
    params: { num_ctx: 8192, temperature: 0.2 }
  explain:
    model: coder
    params: { num_ctx: 8192 }
  summarize:
    model: fast
    params: { num_ctx: 4096 }
```

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENTRYPOINT ["onecard"]
CMD ["--help"]
```

`docker-compose.yml`:

```yaml
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama-models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  onecard:
    build: .
    depends_on: [ollama]
    volumes:
      - ./onecard.yaml:/config/onecard.yaml:ro
      - ./data:/data
    command: ["--help"]

volumes:
  ollama-models:
```

`docker-compose.cpu.yml`:

```yaml
# Overlay for contributors with no NVIDIA card:
#   docker compose -f docker-compose.yml -f docker-compose.cpu.yml up
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices: []
```

`README.md`:

````markdown
# onecard

**A local AI assistant for one small GPU.**

Every local-AI tool assumes you have 24GB of VRAM. On an 8GB card you can hold
*one* 7B model — so you get one generalist, and generalists are worse at every
specific job than a specialist would be.

onecard lets you declare a specialist per task and swaps them for you:

```yaml
tasks:
  code:      { model: coder }
  summarize: { model: fast }
```

```bash
onecard run code "why is this function slow?"
```

It is a **GPU arbiter**, not just a model router: one component owns the card
and decides who holds it, so nothing ever quietly spills to system RAM.

## Quickstart

```bash
docker compose up -d
docker compose run --rm onecard validate --config /config/onecard.yaml
docker compose run --rm onecard run summarize "some long text" --ollama http://ollama:11434
```

No NVIDIA card? Everything still runs, slowly, on CPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Status

Early. The core router, arbiter, and CLI work. Image generation, voice, memory,
and tools are designed but not yet built — see
[the design doc](docs/superpowers/specs/2026-09-08-onecard-design.md).

**No VRAM figure in this repo has been measured on real 8GB hardware yet.** If
you have such a card, correcting the numbers in `profiles/` is the single most
useful contribution you can make.

## License

Apache-2.0
````

- [ ] **Step 4: Run tests to verify they pass**

Run: `ruff check src tests && mypy && pytest -v`
Expected: whole suite green, including 5 packaging tests

- [ ] **Step 5: Commit and push**

```bash
git add Dockerfile docker-compose.yml docker-compose.cpu.yml README.md onecard.yaml profiles tests/test_packaging.py
git commit -m "feat: docker packaging, shipped profile, and README"
git push origin main
```

---

## Self-Review

**Spec coverage.** Each core requirement maps to a task: declared budget and residency tiers (6), fit calculation (3, 8), eviction via `keep_alive` (9), confirmed eviction (6, 7), self-correcting footprints (10), swap-aware scheduling (8, 11), tasks-as-unit with `model`/`steps`/`workflow` exclusivity (2, 4), validate-time overcommit detection (4), streaming and arbitration progress (9, 11, 12), packaging with CPU override (13), profiles with dated headers (13).

Deferred to later plans, each with an explicit loud failure in the core so nothing degrades silently: image generation (`Executor.run` raises `NotImplementedError` on workflow plans — tested), memory, tools, permissions, plugins, voice, `init` wizard, disk-budget enforcement, and the HTTP/OpenAI-compatible API. `permissions`, `tools`, and `memory` already parse in the config schema, so configs written today stay valid when Plan 3 lands.

**Placeholder scan.** No TBDs, no "add error handling", no "similar to Task N". The one artifact not reproduced inline is the Apache-2.0 licence text, with its canonical URL given — reproducing a licence from memory is how licences get subtly corrupted.

**Type consistency.** `Residency`, `GpuConsumer`, `SwapEvent`, `ExecStep`, `ExecPlan`, `Arbiter.claim`, `Arbiter.restore_pinned`, `FootprintStore.get/record/all`, `OllamaConsumer.chat`, and `Executor.run` keep identical signatures everywhere they appear. `estimate_kv_mb(num_ctx, mb_per_1k)` is called positionally and consistently in Tasks 4 and 8. `DEFAULT_FOOTPRINT_MB` lives in one place (`gpu/consumer.py`) and is imported by both `plan.py` and `ollama.py`.

**Ordering fix applied during review.** The original draft had config validation before KV estimation, which `validate.py` imports. Estimation is now Task 3 and validation Task 4, so every task's tests can pass when run in order.
