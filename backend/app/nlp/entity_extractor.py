"""Rule-based entity extraction (order IDs, emails, phones, amounts)."""

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\d{10}(?!\d)")
_AMOUNT = re.compile(r"(?:₹|rs\.?|inr|\$|€|usd|eur)\s?\d+(?:[.,]\d+)?", re.IGNORECASE)
_ORDER_PREFIXED = re.compile(r"\bord[-\s]?(\d{3,10})\b", re.IGNORECASE)
_ORDER_HASH = re.compile(r"#\s?(\d{3,10})\b")
_ORDER_BARE = re.compile(r"(?<![\d-])(\d{5,9})(?![\d-])")


class EntityExtractor:
    def extract(self, text: str) -> dict[str, str]:
        entities: dict[str, str] = {}

        if m := _EMAIL.search(text):
            entities["email"] = m.group(0)
            text = text.replace(m.group(0), " ")
        if m := _PHONE.search(text):
            entities["phone"] = re.sub(r"[\s-]", "", m.group(0))
            text = text.replace(m.group(0), " ")
        if m := _AMOUNT.search(text):
            entities["amount"] = m.group(0).strip()
            text = text.replace(m.group(0), " ")

        for pattern in (_ORDER_PREFIXED, _ORDER_HASH, _ORDER_BARE):
            if m := pattern.search(text):
                entities["order_id"] = f"ORD-{m.group(1)}"
                break
        return entities
