import json
import os
import os.path as osp

from typing import Any, Dict, List


def load_config(file: str) -> Dict:
    """Load a configuration file.

    Arguments:
        file: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
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


def get_active_inputs(model_config: Dict[str, Any]) -> List[str]:
    """Returns the active model inputs based on the config switches."""
    inputs = list(model_config.get('inputs') or [])
    input_enable = model_config.get('input_enable') or {}

    if not input_enable:
        return inputs

    active_inputs = [input_name for input_name in inputs if input_enable.get(input_name, True)]
    if not active_inputs:
        raise ValueError('No active inputs found. Please enable at least one model input.')
    return active_inputs
