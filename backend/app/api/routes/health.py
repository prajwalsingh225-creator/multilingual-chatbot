from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.pipeline import ChatPipeline, get_pipeline

router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health", response_description="Current service status")
def health(pipeline: Annotated[ChatPipeline, Depends(get_pipeline)]) -> dict[str, object]:
    """Liveness probe plus which intent classifier is actually in use."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "languages": settings.SUPPORTED_LANGUAGES,
        "intent_model_loaded": pipeline.models.intent_model_loaded,
        "classifier": "transformer" if pipeline.models.intent_model_loaded else "rules",
    }
