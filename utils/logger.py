from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, output_dir: str | Path | None = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(output_dir / "run.log", mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger

