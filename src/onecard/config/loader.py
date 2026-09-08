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
