from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "train.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """Safely loads and parses the global YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r") as file:
        return yaml.safe_load(file)


CONFIG = load_config()
