"""Resolve context-dependent messages.

Examples:
  "where is my order"  -> asks for the order id, remembers pending_intent=track_order
  "ORD-12345"          -> classifier says fallback, but pending_intent wins -> track_order
  "cancel it"          -> order_id is remembered from earlier turns
"""

from dataclasses import dataclass

from app.conversation.session_manager import Session
from app.nlp.intent_classifier import FALLBACK_INTENT, IntentPrediction

_NOT_CONTINUABLE = {FALLBACK_INTENT, "greeting", "goodbye"}


@dataclass(frozen=True)
class ResolvedContext:
    intent: str
    confidence: float
    entities: dict[str, str]
    is_follow_up: bool


class ContextManager:
    def resolve(
        self, session: Session, prediction: IntentPrediction, new_entities: dict[str, str]
    ) -> ResolvedContext:
        intent = prediction.intent
        is_follow_up = False

        if intent == FALLBACK_INTENT:
            if session.pending_intent:
                # We asked the user for something; treat this message as the answer.
                intent, is_follow_up = session.pending_intent, True
            elif (
                new_entities
                and session.last_intent is not None
                and session.last_intent not in _NOT_CONTINUABLE
            ):
                # A bare entity (e.g. an order id) continues the previous topic.
                intent, is_follow_up = session.last_intent, True

        session.entities = {**session.entities, **new_entities}
        return ResolvedContext(
            intent=intent,
            confidence=prediction.confidence,
            entities=dict(session.entities),
            is_follow_up=is_follow_up,
        )
