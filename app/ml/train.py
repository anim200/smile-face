"""Training logic.

This module knows nothing about HTTP or databases. The CLI script and the web
application both call ``train_from_directory``, so there is exactly one
definition of what training means.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from app.core.constants import CLASS_LABELS, LABEL_SMILING
from app.core.exceptions import InsufficientTrainingDataError, InvalidImageError
from app.ml.artifact import ModelArtifact, make_version
from app.ml.features import load_grayscale
from app.ml.pipeline import RANDOM_STATE, build_pipeline

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

#: Below this per class, a held-out split would be too small to mean anything.
HOLDOUT_MIN_PER_CLASS = 10
#: The absolute floor for fitting a two class model at all.
ABSOLUTE_MIN_PER_CLASS = 3
TEST_SIZE = 0.2


def count_images_per_class(root: Path) -> dict[str, int]:
    """Count usable image files in each class subdirectory of ``root``."""
    return {label: len(_image_paths(root / label)) for label in CLASS_LABELS}


def train_from_directory(root: Path, source: str) -> ModelArtifact:
    """Fit a model from ``root/<label>/*.jpg`` and return the artifact.

    Args:
        root: Directory containing one subdirectory per class label.
        source: Free text recording where the data came from, stored in the
            artifact so a prediction can be traced back to its training data.

    Raises:
        InsufficientTrainingDataError: if either class is too small to fit.
    """
    images, labels = _load_dataset(root)
    counts = {label: int((labels == label).sum()) for label in CLASS_LABELS}
    smallest = min(counts.values())

    if smallest < ABSOLUTE_MIN_PER_CLASS:
        raise InsufficientTrainingDataError(
            f"Need at least {ABSOLUTE_MIN_PER_CLASS} images per class, found {counts}"
        )

    logger.info("Training on %d images %s", len(labels), counts)
    pipeline = build_pipeline()

    if smallest >= HOLDOUT_MIN_PER_CLASS:
        metrics = _fit_with_holdout(pipeline, images, labels)
    else:
        metrics = _fit_with_cross_validation(pipeline, images, labels, smallest)

    metrics["class_counts"] = counts

    return ModelArtifact(
        pipeline=pipeline,
        classes=sorted(CLASS_LABELS),
        version=make_version(),
        trained_at=datetime.now(timezone.utc),
        n_samples=len(labels),
        metrics=metrics,
        source=source,
    )


def _fit_with_holdout(pipeline, images, labels) -> dict:
    """Hold out a stratified test split, score on it, then refit on everything.

    Refitting on the full set is deliberate: the split exists to produce an
    honest estimate, and there is no reason to ship a model trained on 80% of
    the data once that estimate has been taken.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    pipeline.fit(x_train, y_train)
    metrics = _score(y_test, pipeline.predict(x_test), strategy="holdout")
    pipeline.fit(images, labels)
    return metrics


def _fit_with_cross_validation(pipeline, images, labels, smallest: int) -> dict:
    """Score small datasets by cross validation, since a holdout would be noise."""
    folds = min(3, smallest)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    predicted = cross_val_predict(pipeline, images, labels, cv=splitter)
    metrics = _score(labels, predicted, strategy=f"{folds}-fold cross validation")
    pipeline.fit(images, labels)
    return metrics


def _score(y_true, y_predicted, strategy: str) -> dict:
    """Report accuracy alongside minority class F1.

    Accuracy alone is misleading on this dataset: a model that always answers
    "not smiling" scores roughly 0.77 while being useless.
    """
    return {
        "strategy": strategy,
        "accuracy": round(float(accuracy_score(y_true, y_predicted)), 4),
        "f1_smiling": round(
            float(
                f1_score(y_true, y_predicted, pos_label=LABEL_SMILING, zero_division=0)
            ),
            4,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_predicted, labels=sorted(CLASS_LABELS)
        ).tolist(),
        "label_order": sorted(CLASS_LABELS),
    }


def _load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read every class subdirectory into arrays, skipping unreadable files."""
    images: list[np.ndarray] = []
    labels: list[str] = []
    skipped = 0

    for label in CLASS_LABELS:
        for path in _image_paths(root / label):
            try:
                images.append(load_grayscale(path))
            except InvalidImageError:
                skipped += 1
                continue
            labels.append(label)

    if skipped:
        logger.warning("Skipped %d unreadable file(s) under %s", skipped, root)
    if not images:
        raise InsufficientTrainingDataError(f"No readable images found under {root}")

    return np.stack(images), np.asarray(labels)


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )