"""Logging configuration with per-request correlation IDs."""

import contextvars
import logging
import logging.config
import uuid

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(request_id)s | %(message)s"

# Set by the request-id middleware; "-" outside a request (startup, scripts).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every record so log lines can be correlated."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def new_request_id() -> str:
    return uuid.uuid4().hex


def setup_logging(level: str = "INFO", debug: bool = False) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_id": {"()": RequestIdFilter}},
            "formatters": {"default": {"format": _FORMAT}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "filters": ["request_id"],
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": "DEBUG" if debug else level.upper(), "handlers": ["console"]},
            "loggers": {
                # keep noisy libraries quiet
                "sqlalchemy.engine": {"level": "WARNING"},
                "httpx": {"level": "WARNING"},
                "urllib3": {"level": "WARNING"},
                "transformers": {"level": "WARNING"},
            },
        }
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)