"""FastAPI application entry point.  Run:  uv run uvicorn app.main:app --reload"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, health, sessions
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, new_request_id, request_id_var, setup_logging
from app.database.database import init_db
from app.pipeline import build_pipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging(settings.LOG_LEVEL, settings.DEBUG)
    init_db()
    app.state.pipeline = build_pipeline(settings)
    logger.info("%s started (env=%s)", settings.APP_NAME, settings.APP_ENV)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach an X-Request-ID to every request/response and to its log lines."""
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "%s %s %s failed after %.1fms",
                request_id,
                request.method,
                request.url.path,
                (time.perf_counter() - started) * 1000,
            )
            raise
        finally:
            request_id_var.reset(token)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        # Never log bodies or query strings: message text can contain PII.
        logger.info(
            "%s %s %s -> %s in %.1fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "detail": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Same ``{"error", "detail"}`` shape as every other error response."""
        parts = []
        for error in exc.errors():
            location = ".".join(str(piece) for piece in error["loc"] if piece != "body")
            parts.append(f"{location}: {error['msg']}" if location else error["msg"])
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "; ".join(parts) or "request validation failed",
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Never leak a stack trace to the client, especially in production."""
        logger.exception("%s unhandled %s", request.url.path, type(exc).__name__)
        detail = (
            "Internal server error"
            if settings.is_production
            else f"Internal server error: {type(exc).__name__}: {exc}"
        )
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": detail})

    app.include_router(health.router, prefix=settings.API_PREFIX)
    app.include_router(chat.router, prefix=settings.API_PREFIX)
    app.include_router(sessions.router, prefix=settings.API_PREFIX)
    return app


app = create_app()