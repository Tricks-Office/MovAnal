"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Path where to save the configuration.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries.

    Values in override_config take precedence over base_config.

    Args:
        base_config: Base configuration dictionary.
        override_config: Configuration to override base values.

    Returns:
        Merged configuration dictionary.
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration has required fields.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        True if valid, raises ValueError otherwise.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    required_sections = ["video", "preprocessing", "features"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate video settings
    video = config.get("video", {})
    if video.get("width", 0) <= 0 or video.get("height", 0) <= 0:
        raise ValueError("Video width and height must be positive integers")
    if video.get("fps", 0) <= 0:
        raise ValueError("Video fps must be a positive number")

    # Validate preprocessing settings
    preproc = config.get("preprocessing", {})
    resize = preproc.get("resize", {})
    if resize.get("width", 0) <= 0 or resize.get("height", 0) <= 0:
        raise ValueError("Preprocessing resize dimensions must be positive")

    return True


def get_default_config_path() -> Path:
    """Get the path to the default configuration file.

    Returns:
        Path to configs/default.yaml
    """
    project_root = Path(__file__).parent.parent.parent
    return project_root / "configs" / "default.yaml"


def load_config_with_defaults(
    config_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Load configuration with defaults.

    Loads the default configuration and merges with custom config if provided.

    Args:
        config_path: Optional path to custom configuration file.

    Returns:
        Merged configuration dictionary.
    """
    default_config = load_config(get_default_config_path())

    if config_path is not None:
        custom_config = load_config(config_path)
        return merge_configs(default_config, custom_config)

    return default_config


class Config:
    """Configuration wrapper class for easy attribute access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize Config from dictionary.

        Args:
            config_dict: Configuration dictionary.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back to dictionary.

        Returns:
            Configuration as dictionary.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
