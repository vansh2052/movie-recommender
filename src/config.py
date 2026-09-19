"""Single loader for config.yaml. Every path and hyperparameter in the
project is read through this module instead of being hardcoded elsewhere."""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve declared paths relative to the project root so scripts work
    # regardless of the caller's current working directory.
    for key, value in cfg["paths"].items():
        if key.endswith("_dir"):
            cfg["paths"][key] = str(PROJECT_ROOT / value)

    return cfg


CONFIG = load_config()
