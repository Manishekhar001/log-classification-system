import argparse
import logging

import config
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm
from processor_regex import classify_with_regex

logger = logging.getLogger(__name__)


def classify(logs):
    """Classify a list of (source, log_message) tuples.

    Returns:
        list of (label, confidence) tuples.
    """
    results = []
    for source, log_message in logs:
        result = classify_logs(source, log_message)
        results.append(result)
    return results


def classify_logs(source, log_message):
    """Route a single log to the appropriate classifier with fallback.

    Returns:
        (label, confidence) tuple. Returns ("Unclassified", 0.0) on error.
    """
    try:
        # Short-circuit for empty or very short messages
        if not log_message or len(log_message.strip()) < config.MIN_LOG_LENGTH:
            logger.debug("Message too short (len=%d), returning Unclassified", len(log_message or ""))
            return "Unclassified", 0.0

        # Try regex FIRST for ALL sources (cheap, fast, 1.0 confidence)
        label, confidence = classify_with_regex(log_message)
        if label:
            logger.info("Regex matched: %s (conf=%.2f)", label, confidence)
            return label, confidence

        # If source is LegacyCRM, use LLM (it has domain-specific patterns)
        if source == "LegacyCRM":
            if not config.ENABLE_LLM:
                logger.warning("LLM disabled but required for LegacyCRM source, returning Unclassified")
                return "Unclassified", 0.0
            label, confidence = classify_with_llm(log_message)
            logger.info("LegacyCRM -> LLM: %s (conf=%.2f)", label, confidence)
            return label, confidence

        # Non-LegacyCRM: fall back to BERT
        if not config.ENABLE_BERT:
            logger.warning("BERT disabled, returning Unclassified")
            return "Unclassified", 0.0

        label, confidence = classify_with_bert(log_message)
        logger.info("BERT classified: %s (conf=%.2f)", label, confidence)
        return label, confidence

    except Exception as exc:
        logger.error("Classification failed for source=%r msg=%r: %s", source, str(log_message)[:80], exc)
        return "Unclassified", 0.0


def classify_csv(input_file, output_file=None):
    """Classify all logs in a CSV file and write results with confidence scores."""
    import os
    import pandas as pd

    df = pd.read_csv(input_file)

    # Validate required columns exist
    required_cols = ["source", "log_message"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    # Strip whitespace from text columns to avoid mismatches
    for col in required_cols:
        df[col] = df[col].astype(str).str.strip()

    results = classify(list(zip(df["source"], df["log_message"])))
    df["target_label"] = [r[0] for r in results]
    df["confidence"] = [round(r[1], 4) for r in results]

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_classified{ext}"
    df.to_csv(output_file, index=False)
    logger.info("Results written to %s", output_file)
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Log Classification System - Classify system logs using regex, BERT, and LLM."
    )
    parser.add_argument("input_file", nargs="?", help="Path to a CSV file with 'source' and 'log_message' columns")
    parser.add_argument("-o", "--output", help="Output CSV file path (default: derived from input)")
    parser.add_argument("--log-level", default=config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set the logging level")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    if args.input_file:
        # CSV mode
        output = classify_csv(args.input_file, args.output)
        print(f"Classification complete. Output written to: {output}")
        return

    # Demo mode: classify the built-in sample logs
    logger.info("Running demo with built-in sample logs")
    demo_logs = [
        ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
        ("BillingSystem", "User 12345 logged in."),
        ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
        ("AnalyticsEngine", "Backup completed successfully."),
        ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400"),
        ("ModernHR", "Admin access escalation detected for user 9429"),
        ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
        ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
        ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
        ("LegacyCRM", "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"),
    ]

    results = classify(demo_logs)

    print()
    print(f"{'Source':20s} {'Label':25s} {'Conf':6s} Log Message")
    print("-" * 100)
    for (source, msg), (label, confidence) in zip(demo_logs, results):
        print(f"{source:20s} {label:25s} {confidence:.2f}  {msg[:50]}")
    print()
    print(f"Classified {len(results)} logs ("
          f"{sum(1 for _, c in results if c >= 0.5)} high-confidence, "
          f"{sum(1 for l, _ in results if l == 'Unclassified')} unclassified)")


if __name__ == "__main__":
    main()
