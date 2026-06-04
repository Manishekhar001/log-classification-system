import logging
import os
import re
from functools import lru_cache

from groq import Groq

import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Get or create the Groq client. Reads API key lazily on first call."""
    global _client
    if _client is None:
        api_key = config.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key, timeout=config.LLM_TIMEOUT)
    return _client

@lru_cache(maxsize=config.LLM_CACHE_SIZE)
def classify_with_llm(log_msg):
    """Classify a log message using the Groq LLM.

    Returns:
        (label, 1.0) on success, ("Unclassified", 0.0) on error.
    """
    if not config.ENABLE_LLM:
        return "Unclassified", 0.0

    try:
        client = _get_client()
        prompt = f"""Classify the log message into one of these categories:
    (1) Workflow Error, (2) Deprecation Warning.
    If you cannot figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags.
    Log message: {log_msg}"""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT,
        )

        content = chat_completion.choices[0].message.content
        match = re.search(r"<category>(.*)</category>", content, flags=re.DOTALL)
        category = "Unclassified"
        if match:
            category = match.group(1).strip()

        logger.debug("LLM classified: %s -> %s", log_msg[:50], category)
        return category, 1.0

    except Exception as exc:
        logger.error("LLM classification failed: %s", exc)
        return "Unclassified", 0.0

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tests = [
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025",
        "System reboot initiated by user 12345.",
    ]
    for msg in tests:
        label, conf = classify_with_llm(msg)
        print(f"{msg[:60]:60s} -> {label} (conf={conf:.2f})")
