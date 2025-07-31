"""Configuration management for SignalCLI."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


class ConfigurationError(Exception):
    """Configuration related errors."""

    pass


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If config file is invalid or missing
    """
    if config_path is None:
        # Try multiple locations
        possible_paths = [
            Path("config/settings.yaml"),
            Path("/etc/signalcli/settings.yaml"),
            Path.home() / ".signalcli" / "settings.yaml",
            Path(__file__).parent.parent.parent / "config" / "settings.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        else:
            raise ConfigurationError(
                f"No configuration file found in: {[str(p) for p in possible_paths]}"
            )

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ["api", "llm", "vector_store", "rag"]
        missing = [s for s in required_sections if s not in config]
        if missing:
            raise ConfigurationError(f"Missing required config sections: {missing}")

        # Apply environment variable overrides
        config = _apply_env_overrides(config)

        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}")
    except FileNotFoundError:
        raise ConfigurationError(f"Config file not found: {config_path}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config: {e}")


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config."""
    # Map of env vars to config paths
    env_mappings = {
        "SIGNALCLI_API_HOST": ("api", "host"),
        "SIGNALCLI_API_PORT": ("api", "port"),
        "SIGNALCLI_LLM_MODEL_PATH": ("llm", "model_path"),
        "SIGNALCLI_VECTOR_HOST": ("vector_store", "host"),
        "SIGNALCLI_VECTOR_PORT": ("vector_store", "port"),
        "SIGNALCLI_LOG_LEVEL": ("logging", "level"),
    }

    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the right config section
            current = config
            for key in config_path[:-1]:
                current = current.setdefault(key, {})

            # Convert port numbers to int
            if "port" in config_path[-1].lower():
                try:
                    value = int(value)
                except ValueError:
                    continue

            current[config_path[-1]] = value

    return config


def get_config_value(path: str, default: Any = None) -> Any:
    """
    Get a specific configuration value by path.

    Args:
        path: Dot-separated path (e.g., 'api.port')
        default: Default value if path not found

    Returns:
        Configuration value or default
    """
    config = load_config()

    try:
        value = config
        for key in path.split("."):
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


class Config:
    """Configuration wrapper with attribute access."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is None:
            config_dict = load_config()
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute '{name}'")
