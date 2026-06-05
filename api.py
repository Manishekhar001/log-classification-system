"""FastAPI server for the Log Classification System.

Provides HTTP endpoints to classify logs via POST requests.
Run with: uvicorn api:app --reload
"""

import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from classify import classify, classify_logs

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Log Classification API",
    description="Classify system logs using regex, BERT, and LLM fallback strategies.",
    version="1.0.0",
)


# --- Request / Response Models ---

class LogEntry(BaseModel):
    """A single log entry to classify."""
    source: str
    log_message: str


class SingleLogResponse(BaseModel):
    """Response for a single log classification."""
    source: str
    log_message: str
    label: str
    confidence: float


class BatchLogRequest(BaseModel):
    """Batch of log entries to classify."""
    logs: list[LogEntry]


class BatchLogResponse(BaseModel):
    """Response for a batch classification."""
    results: list[SingleLogResponse]
    summary: dict


# --- Endpoints ---

@app.get("/health", summary="Health check")
def health():
    """Return server health status."""
    return {
        "status": "ok",
        "strategies": {
            "regex": config.ENABLE_REGEX,
            "bert": config.ENABLE_BERT,
            "llm": config.ENABLE_LLM,
        },
    }


@app.post("/classify", response_model=SingleLogResponse, summary="Classify a single log")
def classify_single(entry: LogEntry):
    """Classify a single log message by source and message text."""
    if not entry.source or not entry.source.strip():
        raise HTTPException(status_code=400, detail="source must be a non-empty string")
    if not entry.log_message or not entry.log_message.strip():
        raise HTTPException(status_code=400, detail="log_message must be a non-empty string")

    try:
        label, confidence = classify_logs(entry.source, entry.log_message)
        return SingleLogResponse(
            source=entry.source,
            log_message=entry.log_message,
            label=label,
            confidence=round(confidence, 4),
        )
    except Exception as exc:
        logger.error("API classify failed: %s", exc)
        raise HTTPException(status_code=500, detail="Classification failed internally") from exc


@app.post("/classify/batch", response_model=BatchLogResponse, summary="Classify multiple logs")
def classify_batch(request: BatchLogRequest):
    """Classify multiple log messages in a single request."""
    if not request.logs:
        raise HTTPException(status_code=400, detail="No logs provided")

    # Validate all entries
    for i, log in enumerate(request.logs):
        if not log.source or not log.source.strip():
            raise HTTPException(status_code=400, detail=f"Entry {i}: source must be a non-empty string")
        if not log.log_message or not log.log_message.strip():
            raise HTTPException(status_code=400, detail=f"Entry {i}: log_message must be a non-empty string")

    try:
        entries = [(log.source, log.log_message) for log in request.logs]
        results = classify(entries)

        response_logs = [
            SingleLogResponse(
                source=log.source,
                log_message=log.log_message,
                label=label,
                confidence=round(confidence, 4),
            )
            for log, (label, confidence) in zip(request.logs, results)
        ]

        high_conf = sum(1 for _, c in results if c >= 0.5)
        unclassified = sum(1 for l, _ in results if l == "Unclassified")

        summary = {
            "total": len(results),
            "high_confidence": high_conf,
            "unclassified": unclassified,
        }

        return BatchLogResponse(results=response_logs, summary=summary)

    except Exception as exc:
        logger.error("API batch classify failed: %s", exc)
        raise HTTPException(status_code=500, detail="Batch classification failed internally") from exc


# --- Entry point ---

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(levelname)s | %(name)s | %(message)s",
    )
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
