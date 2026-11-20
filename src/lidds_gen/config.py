"""YAML config loading with dotted-key command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        apply_override(cfg, item)
    return cfg


def apply_override(cfg: dict[str, Any], item: str) -> None:
    """Apply ``a.b.c=value`` in place; ``value`` is parsed as YAML (so 3, true, [1,2] work)."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"Override must look like key=value, got {item!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def dump_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)
