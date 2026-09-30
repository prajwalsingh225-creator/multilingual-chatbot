"""Logging configuration with per-request correlation IDs."""

import contextvars
import logging
import sys
import uuid

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(request_id)s | %(message)s"
_HANDLER_TAG = "_app_console_handler"

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
    """Configure logging for the app without disturbing other handlers.

    Deliberately NOT ``logging.config.dictConfig``: that replaces the root logger's
    handlers wholesale, which silently removes handlers installed by the embedding
    process (most visibly pytest's ``caplog`` fixture, making every log assertion pass
    vacuously on empty output). Instead this manages exactly one handler that it
    tagged itself, so repeated calls stay idempotent and foreign handlers survive.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else level.upper())

    for handler in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.addFilter(RequestIdFilter())
    setattr(handler, _HANDLER_TAG, True)
    root.addHandler(handler)

    # Keep noisy third-party libraries quiet.
    for name in ("sqlalchemy.engine", "httpx", "urllib3", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)