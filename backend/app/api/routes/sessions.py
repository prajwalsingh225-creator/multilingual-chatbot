"""Session lifecycle endpoints.

Note: these schemas intentionally avoid ``from __future__ import annotations`` so that
FastAPI evaluates the annotations against real Pydantic models at runtime.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session as DBSession

from app.api.schemas.session import MessageOut, SessionCreate, SessionHistory, SessionOut
from app.core.exceptions import SessionNotFoundError
from app.database.database import get_db
from app.database.repository import ConversationRepository
from app.pipeline import ChatPipeline, get_pipeline

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a session",
    response_description="The new session id",
)
def create_session(
    payload: SessionCreate,
    pipeline: Annotated[ChatPipeline, Depends(get_pipeline)],
    db: Annotated[DBSession, Depends(get_db)],
) -> SessionOut:
    """Create a new conversation session and persist it."""
    session = pipeline.sessions.create(language=payload.language)
    ConversationRepository(db).create_session(session.session_id, language=session.language)
    return SessionOut(
        session_id=session.session_id,
        language=session.language,
        created_at=session.created_at,
    )


@router.get(
    "/{session_id}",
    response_model=SessionHistory,
    summary="Get session transcript",
    response_description="Every persisted message, oldest first",
)
def get_session(
    session_id: str,
    pipeline: Annotated[ChatPipeline, Depends(get_pipeline)],
    db: Annotated[DBSession, Depends(get_db)],
) -> SessionHistory:
    """Return the persisted transcript for a session."""
    repo = ConversationRepository(db)
    record = repo.get_session(session_id)
    if record is None:
        raise SessionNotFoundError(f"Session '{session_id}' not found")
    try:
        pipeline.sessions.get(session_id)
    except SessionNotFoundError:
        # Expired in memory but still persisted: the transcript remains readable.
        pass
    return SessionHistory(
        session_id=session_id,
        messages=[MessageOut.model_validate(m) for m in repo.list_messages(session_id)],
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
    response_description="No content",
)
def delete_session(
    session_id: str,
    pipeline: Annotated[ChatPipeline, Depends(get_pipeline)],
    db: Annotated[DBSession, Depends(get_db)],
) -> Response:
    """Delete a session from memory and the database."""
    repo = ConversationRepository(db)
    deleted = repo.delete_session(session_id)
    try:
        pipeline.sessions.delete(session_id)
    except SessionNotFoundError:
        pass
    if not deleted:
        raise SessionNotFoundError(f"Session '{session_id}' not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)