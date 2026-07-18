import logging
from pathlib import Path

dir = Path("logs")
dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    handlers=[logging.FileHandler(dir / "Pipline.log"), logging.StreamHandler()],
)
