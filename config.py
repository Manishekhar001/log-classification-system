"""
Configuration for the Log Classification System.
All settings can be overridden via environment variables.
"""

import os


# --- LLM Settings ---
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-r1-distill-llama-70b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
LLM_CACHE_SIZE = int(os.environ.get("LLM_CACHE_SIZE", "128"))

# --- BERT Settings ---
BERT_EMBEDDING_MODEL = os.environ.get("BERT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BERT_CONFIDENCE_THRESHOLD = float(os.environ.get("BERT_CONFIDENCE_THRESHOLD", "0.5"))

# --- Strategy Toggles ---
ENABLE_REGEX = os.environ.get("ENABLE_REGEX", "true").lower() in ("true", "1", "yes")
ENABLE_BERT = os.environ.get("ENABLE_BERT", "true").lower() in ("true", "1", "yes")
ENABLE_LLM = os.environ.get("ENABLE_LLM", "true").lower() in ("true", "1", "yes")

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# --- Minimum log message length to attempt classification ---
MIN_LOG_LENGTH = int(os.environ.get("MIN_LOG_LENGTH", "3"))

# --- Groq API Key ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Timeouts ---
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "30"))
