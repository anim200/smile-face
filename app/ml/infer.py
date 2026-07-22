"""Inference against the active model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ml.features import load_grayscale
from app.ml.registry import ModelHolder


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classification result, ready for persistence or display."""

    label: str
    confidence: float
    model_version: str


class SmilePredictor:
    """Classifies a single image using whichever model is currently active."""

    def __init__(self, holder: ModelHolder) -> None:
        self._holder = holder

    def predict(self, image_path: Path) -> Prediction:
        """Classify one image.

        Raises:
            ModelNotAvailableError: if no model is active.
            InvalidImageError: if the file cannot be decoded.
        """
        artifact = self._holder.get()
        batch = load_grayscale(image_path)[np.newaxis, ...]

        label = str(artifact.pipeline.predict(batch)[0])
        probabilities = artifact.pipeline.predict_proba(batch)[0]
        confidence = float(np.max(probabilities))

        return Prediction(
            label=label,
            confidence=round(confidence, 4),
            model_version=artifact.version,
        )