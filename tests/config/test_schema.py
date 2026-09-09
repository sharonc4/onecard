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
