"""Configuration loading with explicit, reproducible overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping and reject ambiguous top-level values."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("PyYAML is required to load experiment configurations") from error
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_merged(*paths: str | Path) -> dict[str, Any]:
    """Load YAML files from general to specific."""
    result: dict[str, Any] = {}
    for path in paths:
        result = deep_merge(result, load_yaml(path))
    return result


def resolve_path(value: str | Path, repository_root: str | Path = ".") -> Path:
    """Resolve a configured relative path against a caller-selected root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(repository_root) / path).resolve()
