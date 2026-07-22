"""The on-disk model artifact and its validation rules.

A bare pickled estimator carries no provenance, so the artifact is a small
record: the fitted pipeline plus everything needed to answer "which model
produced this prediction, and can I trust it?".
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from app.core.constants import CLASS_LABELS
from app.core.exceptions import InvalidModelArtifactError

ARTIFACT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class ModelArtifact:
    """Everything persisted alongside the fitted pipeline."""

    pipeline: Pipeline
    classes: list[str]
    version: str
    trained_at: datetime
    n_samples: int
    metrics: dict[str, Any]
    source: str
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    @property
    def accuracy(self) -> float | None:
        value = self.metrics.get("accuracy")
        return float(value) if value is not None else None


def make_version(now: datetime | None = None) -> str:
    """Return a sortable, filesystem safe version string."""
    moment = now or datetime.now(timezone.utc)
    return moment.strftime("%Y%m%dT%H%M%S")


def dump_artifact(artifact: ModelArtifact, destination: Path) -> None:
    """Write the artifact atomically so a reader never sees a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def load_artifact(path: Path) -> ModelArtifact:
    """Read and fully validate an artifact from disk.

    Raises:
        InvalidModelArtifactError: if the file is missing, unreadable, or fails
            any structural or behavioural check.
    """
    if not path.exists():
        raise InvalidModelArtifactError(f"Model file does not exist: {path}")

    try:
        with path.open("rb") as handle:
            loaded = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is fatal here
        raise InvalidModelArtifactError(f"Cannot unpickle {path}: {exc}") from exc

    validate_artifact(loaded)
    return loaded


def validate_artifact(candidate: Any) -> None:
    """Assert that an object is a usable artifact.

    Structural checks catch a corrupt or foreign file. The smoke test catches
    the more dangerous case: an object that deserialises perfectly and is still
    not a working model.
    """
    if not isinstance(candidate, ModelArtifact):
        raise InvalidModelArtifactError(
            f"Expected ModelArtifact, found {type(candidate).__name__}"
        )

    if candidate.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise InvalidModelArtifactError(
            f"Unsupported artifact schema {candidate.schema_version}"
        )

    if sorted(candidate.classes) != sorted(CLASS_LABELS):
        raise InvalidModelArtifactError(
            f"Artifact classes {candidate.classes} do not match {list(CLASS_LABELS)}"
        )

    if not isinstance(candidate.pipeline, Pipeline):
        raise InvalidModelArtifactError("Artifact does not contain a Pipeline")

    try:
        check_is_fitted(candidate.pipeline.named_steps["classifier"])
    except (NotFittedError, KeyError) as exc:
        raise InvalidModelArtifactError(f"Pipeline is not fitted: {exc}") from exc

    _smoke_test(candidate)


def _smoke_test(artifact: ModelArtifact) -> None:
    """Push one blank image through the pipeline end to end.

    Nearly free, and it proves the artifact is callable rather than merely
    well shaped.
    """
    blank = np.zeros((1, 64, 64), dtype=np.float32)
    try:
        prediction = artifact.pipeline.predict(blank)
    except Exception as exc:  # noqa: BLE001 - surface the real reason to the caller
        raise InvalidModelArtifactError(f"Smoke test failed: {exc}") from exc

    if str(prediction[0]) not in CLASS_LABELS:
        raise InvalidModelArtifactError(
            f"Smoke test produced unknown label {prediction[0]!r}"
        )