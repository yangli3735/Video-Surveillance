"""
utils/logger.py
Centralised logging setup for the entire application.
Every module obtains its logger via:  from utils.logger import get_logger
                                       log = get_logger(__name__)
"""
import logging
import logging.handlers
import os
import sys

from config import cfg


def _setup_root_logger() -> None:
    log_cfg = cfg.get("logging", {})
    level_name: str = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir: str = log_cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "surveillance.log")
    max_bytes: int = log_cfg.get("max_bytes", 5 * 1024 * 1024)
    backup_count: int = log_cfg.get("backup_count", 3)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# Run once on import
_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call this at module level)."""
    return logging.getLogger(name)
