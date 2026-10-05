"""
Centralized structured logger for CogniDoc microservices and gateway.
Provides uniform formatting, log-level control, timestamping, and optional ANSI colorization.
"""
import logging
import os
import sys
from typing import Optional

from shared.config import settings

# ANSI color codes for terminal logging
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding color to log levels on interactive terminals."""

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        # Duplicate record so we don't mutate for other handlers
        record_copy = logging.makeLogRecord(record.__dict__)
        record_copy.levelname = f"{color}{record.levelname:<7}{reset}"
        return super().format(record_copy)


def setup_logger(service_name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a root-level or service-level logger.
    Call once at application startup in each service's main.py.
    """
    log_level_str = (level or getattr(settings, "LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        fmt = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        # Check if terminal supports color (TTY or FORCE_COLOR)
        if sys.stdout.isatty() or os.environ.get("FORCE_COLOR"):
            formatter = ColoredFormatter(fmt=fmt, datefmt=datefmt)
        else:
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Silence excessively noisy 3rd-party loggers
    for noisy in ("urllib3", "httpcore", "httpx", "chromadb", "posthog", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(service_name)
    logger.setLevel(log_level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger using the centralized CogniDoc logging configuration."""
    return logging.getLogger(name)
