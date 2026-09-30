"""Measure the intent classifier's fallback rate on deliberately off-topic input.

    uv run --group training python -m training.offtopic_report [--model-dir PATH]

Twenty sentences per supported language that are clearly NOT one of the seven intents.
A healthy classifier must return the ``fallback`` intent for these at the production
confidence threshold, instead of confidently picking something like ``track_order``.

Writes no state; it is a diagnostic, not part of the app.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.nlp.intent_classifier import FALLBACK_INTENT, IntentClassifier
from app.nlp.preprocessor import Preprocessor

OFF_TOPIC: dict[str, list[str]] = {
    "en": [
        "what is the capital of France",
        "tell me a joke about penguins",
        "how tall is the Eiffel Tower",
        "who won the football world cup in 1998",
        "what time does the sun set today",
        "explain photosynthesis to a child",
        "recommend a good science fiction novel",
        "how do i boil an egg",
        "what is 17 times 3",
        "write a haiku about rain",
        "who painted the mona lisa",
        "how far is the moon from earth",
        "what is machine learning",
        "tell me about the roman empire",
        "how do i tie a necktie",
        "what is the boiling point of water",
        "give me a riddle",
        "when was the internet invented",
        "what is the largest ocean",
        "describe the taste of mango",
    ],
    "hi": [
        "फ्रांस की राजधानी क्या है",
        "सीने वाले पर एक चुटकुला सुनाओ",
        "एफिल टॉवर कितना ऊँचा है",
        "1998 में फुटबॉल विश्व कप कौन जीता",
        "आज सूर्यास्त कितने बजे होगा",
        "बच्चों को फोटोसिंथेसिस समझाओ",
        "एक अच्छा साइंस फिक्शन उपन्यास बताओ",
        "अंडा कैसे उबालें",
        "17 गुणा 3 कितना है",
        "बारिश पर एक हायकू लिखो",
        "मोना लिसा किसने बनाई",
        "चाँद पृथ्वी से कितनी दूर है",
        "मशीन लर्निंग क्या है",
        "रोमन साम्राज्य के बारे में बताओ",
        "टाई कैसे बाँधें",
        "पानी का क्वथनांक क्या है",
        "एक पहेली सुनाओ",
        "इंटरनेट कब खोजा गया",
        "सबसे बड़ा महासागर कौन सा है",
        "आम का स्वाद कैसा होता है",
    ],
    "es": [
        "cual es la capital de francia",
        "cuentame un chiste sobre pinguinos",
        "que tan alto es la torre eiffel",
        "quien gano el mundial de futbol en 1998",
        "a que hora se pone el sol hoy",
        "explicame la fotosintesis a un nino",
        "recomienda una buena novela de ciencia ficcion",
        "como hiervo un huevo",
        "cuanto es 17 por 3",
        "escribe un haiku sobre la lluvia",
        "quien pinto la mona lisa",
        "a que distancia esta la luna de la tierra",
        "que es el aprendizaje automatico",
        "cuentame sobre el imperio romano",
        "como hago un nudo de corbata",
        "cual es el punto de ebullicion del agua",
        "dame un acertijo",
        "cuando se invento internet",
        "cual es el oceano mas grande",
        "describe el sabor del mango",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    from app.models.model_loader import load_intent_model

    transformer = load_intent_model(args.model_dir or settings.MODEL_DIR)
    threshold = settings.INTENT_CONFIDENCE_THRESHOLD
    classifier = IntentClassifier(threshold, transformer, None)
    pre = Preprocessor()

    print(f"threshold={threshold} classifier={'transformer' if transformer else 'rules'}")
    overall_total = 0
    overall_fallback = 0
    for language, sentences in OFF_TOPIC.items():
        intents: Counter[str] = Counter()
        fallback = 0
        misses: list[tuple[str, str, float]] = []
        for sentence in sentences:
            prediction = classifier.predict(pre.process(sentence))
            intents[prediction.intent] += 1
            if prediction.intent == FALLBACK_INTENT:
                fallback += 1
            else:
                misses.append((sentence, prediction.intent, prediction.confidence))
        total = len(sentences)
        overall_total += total
        overall_fallback += fallback
        print(f"\n{language}: fallback {fallback}/{total} = {fallback / total:.2f}")
        print(f"  intents returned: {dict(intents.most_common())}")
        for sentence, intent, confidence in misses:
            print(f"  MISS {intent:<16} {confidence:.3f}  {sentence!r}")

    rate = overall_fallback / overall_total
    print(f"\noverall fallback rate: {overall_fallback}/{overall_total} = {rate:.4f}")
    payload: dict[str, Any] = {"threshold": threshold, "overall": rate}
    print(json.dumps(payload))


if __name__ == "__main__":
    main()