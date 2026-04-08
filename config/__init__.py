"""
config/__init__.py
Loads and exposes the YAML configuration as a plain dict/namespace.
Usage:  from config import cfg
        model_path = cfg["inference"]["model_path"]
"""
import os
import yaml


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")


def _load() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


cfg: dict = _load()


def reload():
    """Hot-reload configuration at runtime (useful for testing)."""
    global cfg
    cfg = _load()
