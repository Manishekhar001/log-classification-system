import logging
import re

import config

logger = logging.getLogger(__name__)

REGEX_PATTERNS = {
    r"User (User ?)?\d+ logged (in|out)\.": "User Action",
    r"Backup (started|ended) at .*": "System Notification",
    r"Backup completed successfully\.": "System Notification",
    r"System updated to version .*": "System Notification",
    r"File .* uploaded successfully by user .*": "System Notification",
    r"Disk cleanup completed successfully\.": "System Notification",
    r"System reboot initiated by user .*": "System Notification",
    r"Account with ID .* created by .*": "User Action",
}


def classify_with_regex(log_message):
    """Classify a log message using regex patterns.

    Returns:
        (label or None, confidence) where confidence is 1.0 if matched.
    """
    if not config.ENABLE_REGEX:
        return None, 0.0

    try:
        for pattern, label in REGEX_PATTERNS.items():
            if re.search(pattern, log_message):
                logger.debug("Regex matched: %s -> %s", pattern, label)
                return label, 1.0
    except re.error as exc:
        logger.error("Regex syntax error in pattern %r: %s", pattern, exc)

    return None, 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(classify_with_regex("Backup completed successfully."))
    print(classify_with_regex("Account with ID 1234 created by User1."))
    print(classify_with_regex("User User685 logged out."))
    print(classify_with_regex("User 12345 logged in."))
    print(classify_with_regex("Hey Bro, chill ya!"))
