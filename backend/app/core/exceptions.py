"""Application-level exceptions. main.py maps AppError to a JSON response."""


class AppError(Exception):
    status_code: int = 500
    code: str = "app_error"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.__class__.__doc__ or "Application error"
        super().__init__(self.message)


class ConfigurationError(AppError):
    """Invalid or missing configuration."""

    code = "configuration_error"


class ModelLoadError(AppError):
    """The intent model exists but could not be loaded."""

    code = "model_load_error"


class EmptyMessageError(AppError):
    """Message must not be empty."""

    status_code = 422
    code = "empty_message"


class SessionNotFoundError(AppError):
    """Session not found."""

    status_code = 404
    code = "session_not_found"
