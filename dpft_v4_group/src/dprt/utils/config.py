import json
import os
import os.path as osp

from typing import Any, Dict


def load_config(file: str) -> Dict:
    """Load a configuration file.

    Arguments:
        file: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    if osp.isdir(file):
        raise IsADirectoryError(
            f"Expected a config file path, but got a directory: {file}"
        )
    if not osp.isfile(file):
        raise FileNotFoundError(f"Config file does not exist: {file}")

    with open(file) as f:
        config = json.load(f)

    return config


def loads_config(file: str) -> Dict:
    """Load a configuration file.

    Arguments:
        file: A serialized json file.

    Returns:
        Configuration dictionary.
    """
    config = json.loads(file)

    return config


def save_config(config: Dict[str, Any], filename: str) -> None:
    """Saves configuration to file.

    Arguments:
        config: Configuration dictionary.
        filename: Destination filename (path).
    """
    # Create destination directory (if none existend)
    os.makedirs(osp.dirname(filename), exist_ok=True)

    # Save configuration
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
