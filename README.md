# Log Classification System

A multi-strategy log classification pipeline that categorizes system logs from multiple sources using a tiered fallback approach: **Regex → BERT (ML) → LLM (Groq)**.

Each classifier returns a `(label, confidence)` tuple, and the pipeline gracefully degrades — if one strategy fails, it falls through to the next, or returns `"Unclassified"` with confidence `0.0`.

---

## Classification Pipeline

```
                    ┌──────────────┐
                    │  Input Log   │
                    │ (source, msg)│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Source ==    │
                    │ LegacyCRM?   │
                    └──┬───┬───────┘
                  NO   │   │ YES
              ┌────────┘   └────────┐
              ▼                     ▼
      ┌───────────────┐   ┌────────────────────┐
      │  1. Regex     │   │  3. LLM (Groq)     │
      │  (Fast match) │   │  (Deep understand)  │
      └───────┬───────┘   └────────────────────┘
              │ matched?              │
         YES  │      NO               │
         ┌────┘   ┌──┘               │
         ▼        ▼                   ▼
     ┌──────┐ ┌───────────────┐  ┌──────────────┐
     │Label │ │ 2. BERT + LR │  │  Workflow    │
     │      │ │  (ML fallback)│  │  Error /     │
     │      │ └──────┬───────┘  │  Deprecation │
     │      │        │          │  Warning     │
     └──────┘   ┌────┘          └──────────────┘
                ▼
         ┌──────────────┐
         │ Label or     │
         │ Unclassified │
         └──────────────┘
```

### Categories

| Category | Classifier | Description |
|---|---|---|
| `User Action` | Regex | User login/logout, account creation |
| `System Notification` | Regex | Backups, system updates, reboots, disk cleanup |
| `HTTP Status` | BERT | HTTP response codes from server logs |
| `Critical Error` | BERT | Kernel panics, RAID failures, system crashes |
| `Error` | BERT | General errors (replication, email, data) |
| `Security Alert` | BERT | Unauthorized access, brute force, privilege escalation |
| `Resource Usage` | BERT | Memory, disk, CPU usage reports |
| `Workflow Error` | LLM | LegacyCRM workflow/process failures |
| `Deprecation Warning` | LLM | LegacyCRM deprecated feature notices |
| `Unclassified` | Fallback | When no strategy produces a confident result |

### Confidence Scores

Every classification returns a `(label, confidence)` pair:

| Strategy | Confidence | Meaning |
|---|---|---|
| Regex | `1.0` | Exact pattern match (deterministic) |
| BERT | `0.0` – `1.0` | Model probability for the predicted class |
| BERT (low conf) | `< 0.5` | Returns `"Unclassified"` with the actual probability |
| LLM | `1.0` | Successfully classified by the LLM |
| Any (error) | `0.0` | Strategy failed — graceful fallback to `"Unclassified"` |

---

## Project Structure

```
├── classify.py              # Main orchestrator / CLI entry point
├── config.py                # Centralized configuration (env var overrides)
├── processor_regex.py       # Stage 1: Regex-based classification
├── processor_bert.py        # Stage 2: BERT embeddings + Logistic Regression
├── processor_llm.py         # Stage 3: LLM-based classification (Groq API)
├── models/
│   └── log_classifier.joblib # Pre-trained Logistic Regression model
├── Training/
│   ├── Dataset/
│   │   └── synthetic_logs.csv  # 2,410 labeled synthetic log entries
│   └── log_classification.ipynb # Notebook for training the BERT classifier
├── requirements.txt         # Python dependencies with version constraints
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (only needed for LegacyCRM logs)

### Virtual Environment

```bash
# Create
python -m venv venv

# Activate
# Windows (CMD):        venv\Scripts\activate
# Windows (PowerShell):  venv\Scripts\Activate.ps1
# macOS / Linux:         source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Environment Variables

```bash
# Required for LLM classification (LegacyCRM logs):
export GROQ_API_KEY=gsk_your_api_key_here

# Optional overrides (see Configuration section below):
export LLM_MODEL="deepseek-r1-distill-llama-70b"
export BERT_CONFIDENCE_THRESHOLD=0.5
export LOG_LEVEL=INFO
```

---

## Configuration

All settings in `config.py` can be overridden via environment variables.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key for LLM classification |
| `LLM_MODEL` | `deepseek-r1-distill-llama-70b` | Groq model to use |
| `LLM_TEMPERATURE` | `0.5` | Temperature for LLM responses |
| `LLM_CACHE_SIZE` | `128` | Max number of cached LLM responses |
| `BERT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |
| `BERT_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for BERT predictions |
| `ENABLE_REGEX` | `true` | Set to `false` to skip regex classification |
| `ENABLE_BERT` | `true` | Set to `false` to skip BERT classification |
| `ENABLE_LLM` | `true` | Set to `false` to skip LLM classification |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Usage

### CLI — Classify a CSV file

```bash
# Classify a CSV with source and log_message columns
python classify.py logs.csv

# Specify a custom output path
python classify.py logs.csv -o results.csv

# Enable debug logging
python classify.py logs.csv --log-level DEBUG
```

The output CSV includes two additional columns:
- `target_label` — the predicted category
- `confidence` — the confidence score (0.0 – 1.0)

### CLI — Run the Demo

```bash
python classify.py
```

Runs the built-in demo with 10 sample logs, displaying a formatted table with source, label, confidence, and log message.

### Python API

```python
from classify import classify

logs = [
    ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
    ("BillingSystem", "User 12345 logged in."),
    ("LegacyCRM", "Case escalation for ticket ID 7324 failed."),
]

results = classify(logs)  # Returns list of (label, confidence) tuples
for (source, msg), (label, confidence) in zip(logs, results):
    print(f"{source} → {label} (conf={confidence:.2f})")
```

### Classify a CSV from Python

```python
from classify import classify_csv

# Auto-generated output name (input_classified.csv)
classify_csv("logs.csv")

# Custom output path
classify_csv("logs.csv", output_file="results.csv")
```

### Run Individual Processors

```bash
python processor_regex.py     # Regex tests (no deps needed)
python processor_bert.py      # BERT tests (requires sentence-transformers)
python processor_llm.py       # LLM tests (requires GROQ_API_KEY)
```

---

## Error Handling & Resilience

The pipeline is designed to never crash on a single bad log:

- **Regex errors**: Caught and logged, returns `(None, 0.0)`
- **BERT errors**: Model loading failures, encoding errors, or prediction errors → returns `("Unclassified", 0.0)`
- **LLM errors**: API timeouts, rate limits, invalid responses → returns `("Unclassified", 0.0)`
- **Orchestration errors**: Any unexpected exception in `classify_logs()` is caught globally and returns `("Unclassified", 0.0)`

You can disable any strategy at any time by setting the corresponding environment variable to `false`. This is useful for testing or if a dependency is unavailable.

```bash
export ENABLE_BERT=false   # Skip BERT, only use regex (then Unclassified)
export ENABLE_LLM=false    # Mark all LegacyCRM logs as Unclassified
```

---

## Logging

Logs are structured with level, module, and message:

```
INFO  | __main__ | Regex matched: System Notification (conf=1.00)
INFO  | __main__ | BERT classified: HTTP Status (conf=0.97)
INFO  | __main__ | LegacyCRM -> LLM: Workflow Error (conf=1.00)
ERROR | processor_llm | LLM classification failed: Rate limit exceeded
```

Set the log level via `--log-level` on the CLI or the `LOG_LEVEL` env var.

---

## LLM Caching

The `classify_with_llm` function uses `@functools.lru_cache` to avoid redundant API calls. When the same log message is classified twice, the cached result is returned instantly. The cache size is configured via `LLM_CACHE_SIZE` (default: 128 entries).

---

## Training

The BERT-based classifier was trained in `Training/log_classification.ipynb`:

1. Loads and explores 2,410 synthetic log entries from 6 source systems
2. Clusters similar log messages using DBSCAN on sentence embeddings (136 clusters)
3. Separates regex-classifiable (500 entries) and LegacyCRM (7 entries) — not trained on
4. Trains a Logistic Regression model on sentence embeddings of the remaining 1,903 logs
5. Achieves **~99% weighted accuracy** on held-out test data

To retrain:

```bash
cd Training
pip install jupyter sentence-transformers scikit-learn pandas joblib
jupyter notebook log_classification.ipynb
```

---

## Notes

- On first run, `processor_bert.py` downloads the `all-MiniLM-L6-v2` embedding model (~80MB) from Hugging Face.
- The `sentence-transformers` library requires a compatible version of `huggingface_hub` — the pinned versions in `requirements.txt` ensure compatibility.
- The LLM processor uses `deepseek-r1-distill-llama-70b` via Groq. You can switch models by setting the `LLM_MODEL` environment variable.
