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

# Which slot each pending intent is waiting for. Used to recognise a slot answer even
# when the classifier confidently guessed something else for a bare value like
# "ORD-12345" (the fine-tuned model scores that as "goodbye" at 0.56, just over the
# 0.55 threshold, which used to lose the pending intent entirely).
_PENDING_SLOT = {
    "track_order": "order_id",
    "cancel_order": "order_id",
    "refund": "order_id",
    "payment_issue": "order_id",
}


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

        pending = session.pending_intent
        pending_slot = _PENDING_SLOT.get(pending) if pending else None

        if pending and pending_slot and new_entities.get(pending_slot):
            # The bot asked for a slot and the user just supplied it. Honour the pending
            # intent regardless of what the classifier said, so a bare "ORD-12345" fills
            # the slot instead of being answered as whatever it scored highest.
            intent, is_follow_up = pending, True
        elif intent == FALLBACK_INTENT:
            if pending:
                # We asked the user for something; treat this message as the answer.
                intent, is_follow_up = pending, True
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
