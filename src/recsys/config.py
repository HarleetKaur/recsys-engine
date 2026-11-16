import os
from pathlib import Path

import yaml

# Repo root. Relative paths in the config are resolved against it, so commands work from any directory.
ROOT = Path(os.environ.get("RECSYS_HOME") or Path(__file__).resolve().parents[2])

_PATH_KEYS = [("data", "raw_dir"), ("data", "processed_dir"), ("serving", "log_dir"),
              ("paths", "artifacts"), ("paths", "reports"), ("paths", "triton_repo")]


def _absolute(p):
    p = Path(p)
    return str(p if p.is_absolute() else ROOT / p)


def apply_overrides(cfg, items):
    """items like ['retrieval.epochs=10', 'data.dataset=ml-latest-small']; values are parsed as YAML."""
    for item in items or []:
        key, _, raw = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(path=None, overrides=None):
    path = path or os.environ.get("RECSYS_CONFIG") or ROOT / "configs" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, overrides)
    for section, key in _PATH_KEYS:
        cfg[section][key] = _absolute(cfg[section][key])
    return cfg
