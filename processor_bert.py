import logging
import os
import threading

import joblib

import config

logger = logging.getLogger(__name__)

_model_embedding = None
_model_classification = None
_load_lock = threading.Lock()


def _load_models():
    """Lazy-load the embedding and classification models on first use.

    Thread-safe: uses a lock to prevent duplicate loading under concurrent requests.
    """
    global _model_embedding, _model_classification

    if _model_embedding is not None and _model_classification is not None:
        return

    with _load_lock:
        # Double-checked locking: models may have been loaded while waiting for the lock
        if _model_embedding is None:
            logger.info("Loading embedding model: %s", config.BERT_EMBEDDING_MODEL)
            from sentence_transformers import SentenceTransformer
            _model_embedding = SentenceTransformer(config.BERT_EMBEDDING_MODEL)

        if _model_classification is None:
            model_path = os.path.join(os.path.dirname(__file__), "models", "log_classifier.joblib")
            logger.info("Loading classification model: %s", model_path)
            _model_classification = joblib.load(model_path)


def classify_with_bert(log_message):
    """Classify a log message using BERT embeddings + Logistic Regression.

    Returns:
        (label, confidence). Returns ("Unclassified", 0.0) on error
        or if confidence is below the threshold.
    """
    if not config.ENABLE_BERT:
        return "Unclassified", 0.0

    try:
        _load_models()

        embeddings = _model_embedding.encode([log_message])
        probabilities = _model_classification.predict_proba(embeddings)
        confidence = max(probabilities[0])

        if confidence < config.BERT_CONFIDENCE_THRESHOLD:
            logger.debug("BERT confidence %.2f below threshold %.2f, returning Unclassified",
                         confidence, config.BERT_CONFIDENCE_THRESHOLD)
            return "Unclassified", confidence

        prediction_label = _model_classification.predict(embeddings)[0]
        logger.debug("BERT classified: %s -> %s (conf=%.2f)", log_message[:50], prediction_label, confidence)
        return prediction_label, confidence

    except Exception as exc:
        logger.error("BERT classification failed: %s", exc)
        return "Unclassified", 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer",
    ]
    for log in logs:
        label, conf = classify_with_bert(log)
        print(f"{log[:60]:60s} -> {label} (conf={conf:.2f})")
