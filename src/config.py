"""Configuration loader for the RAG pipeline."""

from pathlib import Path

import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
