from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = config
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    i = 0
    while i < len(overrides):
        token = overrides[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected override token: {token}")
        key_value = token[2:]
        if "=" in key_value:
            key, value = key_value.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(overrides):
                raise ValueError(f"Missing value for override: {token}")
            key = key_value
            value = overrides[i + 1]
            i += 2
        set_nested(resolved, key, parse_scalar(value))
    return resolved


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def require(config: dict[str, Any], dotted_key: str) -> Any:
    sentinel = object()
    value = get_config_value(config, dotted_key, sentinel)
    if value is sentinel:
        raise KeyError(f"Missing required config key: {dotted_key}")
    return value
