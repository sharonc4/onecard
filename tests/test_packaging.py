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
