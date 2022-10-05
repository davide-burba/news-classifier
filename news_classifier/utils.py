import os
from typing import Dict, Any
from datetime import datetime as dt
import yaml


def get_time() -> str:
    """Get current time.

    Returns:
        Current UTC time as string in format `%Y-%m-%d_%Hh%Mm%Ss`.
    """
    return dt.utcnow().strftime("%Y-%m-%d_%Hh%Mm%Ss")


def build_output_dir(path_dir: str) -> str:
    """Build UTC timestamped subdir in path_dir, return path (str)."""
    path_dir = f"{path_dir}/run_{get_time()}"
    os.makedirs(path_dir)
    return path_dir


def load_config(config_path: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Load config file from config_path.

    Args:
        config_path: The path to a yaml config file. If None, use the default config.
            Otherwise, its values override values in default_config.
            First level nested dictionaries are updated.
        default_config: A default config dictionary.

    Returns
        The config dictionary.
    """
    config = default_config.copy()
    if config_path is not None:
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f)
        for k, v in custom_config.items():
            if isinstance(v, dict):
                config[k].update(custom_config[k])
            else:
                config[k] = custom_config[k]
    return config
