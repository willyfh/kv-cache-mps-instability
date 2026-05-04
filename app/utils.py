"""Shared utility functions for application runtime configuration."""

import os
from pathlib import Path

import yaml


def load_config(config_path=None):
    """Load YAML configuration from explicit path, env var, or default file."""
    env_config_path = os.getenv("APP_CONFIG_PATH")
    resolved_path = Path(config_path or env_config_path or Path(__file__).resolve().parent.parent / "configs" / "config.yaml")
    with resolved_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}