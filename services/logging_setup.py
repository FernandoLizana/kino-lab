from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import Config


def setup_logging(app=None) -> logging.Logger:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("kino_platform")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        Config.LOG_DIR / "app.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(stream)

    if app is not None:
        app.logger.handlers = logger.handlers
        app.logger.setLevel(logging.INFO)
    return logger
