"""Shared helpers: config loading, logging, parquet I/O. All PROVIDED."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# SYNTAX: Path(__file__) is this file; .resolve().parents[1] walks up to the project root.
ROOT = Path(__file__).resolve().parents[1]


# PROVIDED
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml into a nested dict."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as f:  # SYNTAX: `with` closes the file even if an error happens
        return yaml.safe_load(f)


# PROVIDED
def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """cfg_get(cfg, "trading.min_edge") instead of cfg["trading"]["min_edge"]."""
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# PROVIDED
def env_key(cfg: dict, name: str) -> str:
    """Read an API key from the environment variable named in config.api_keys."""
    var = cfg_get(cfg, f"api_keys.{name}")
    value = os.environ.get(var or "", "")
    if not value:
        raise RuntimeError(f"Set the environment variable {var} (see config.yaml api_keys).")
    return value


# PROVIDED
def data_path(cfg: dict, kind: str, filename: str) -> Path:
    """Absolute path inside data/raw or data/processed; creates the folder."""
    folder = ROOT / cfg_get(cfg, f"paths.data_{kind}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


# PROVIDED
def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# PROVIDED
def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# PROVIDED
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)
