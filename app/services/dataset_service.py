"""Staging area for training uploads.

Uploads arrive one class at a time, because the form carries a single label.
They accumulate here until both classes are present, at which point training
can run. The whole directory is removed once a run completes.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.core.constants import CLASS_LABELS

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class DatasetService:
    """Owns the contents of the staging directory."""

    def __init__(self, staging_dir: Path) -> None:
        self._staging_dir = staging_dir

    @property
    def root(self) -> Path:
        return self._staging_dir

    def directory_for(self, label: str) -> Path:
        """Where images of ``label`` are staged."""
        if label not in CLASS_LABELS:
            raise ValueError(f"Unknown label {label!r}")
        return self._staging_dir / label

    def counts(self) -> dict[str, int]:
        """How many images are staged per class."""
        return {label: len(self._files(label)) for label in CLASS_LABELS}

    def total(self) -> int:
        return sum(self.counts().values())

    def is_ready(self, minimum_per_class: int) -> bool:
        """True when every class has enough images to fit a model."""
        counts = self.counts()
        return all(counts[label] >= minimum_per_class for label in CLASS_LABELS)

    def clear(self) -> None:
        """Remove every staged image, satisfying the delete-after-training rule."""
        if self._staging_dir.exists():
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        for label in CLASS_LABELS:
            (self._staging_dir / label).mkdir(parents=True, exist_ok=True)
        logger.info("Staging cleared")

    def _files(self, label: str) -> list[Path]:
        directory = self._staging_dir / label
        if not directory.is_dir():
            return []
        return [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]