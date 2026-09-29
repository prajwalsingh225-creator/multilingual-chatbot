from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.database.database import get_db
from app.pipeline import ChatPipeline, get_pipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    pipeline: Annotated[ChatPipeline, Depends(get_pipeline)],
    db: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    return pipeline.process(payload.message, payload.session_id, db)
