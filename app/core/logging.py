"""Logging setup.

One call at startup; every module then uses ``logging.getLogger(__name__)``.
"""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Noisy at INFO and adds nothing during normal operation.
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)