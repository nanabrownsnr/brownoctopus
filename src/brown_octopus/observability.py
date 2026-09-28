"""Small structured logging helpers with no logging dependency."""

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event_data"):
            payload["data"] = record.event_data
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    # Libraries should configure their verbosity, not install output handlers
    # or write to stdout. Hosts can attach a handler to this namespace.
    logging.getLogger("brown_octopus").setLevel(level.upper())
