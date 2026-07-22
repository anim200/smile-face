"""The model registry: a single active artifact on disk.

Exactly one model file exists at a time. A training run writes a new file
beside it and only replaces the active one after validation passes, so a run
that fails leaves the previous model serving untouched.

``Path.replace`` is atomic within a filesystem, which means a request arriving
mid-swap reads either the old complete file or the new complete file, never a
half-written one.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.core.exceptions import ModelNotAvailableError
from app.ml.artifact import ModelArtifact, dump_artifact, load_artifact

logger = logging.getLogger(__name__)

MODEL_FILENAME = "smile_clf.pkl"

_promote_lock = threading.Lock()


class ModelRegistry:
    """Owns the lifecycle of the active model file."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir

    @property
    def model_path(self) -> Path:
        return self._model_dir / MODEL_FILENAME

    def exists(self) -> bool:
        return self.model_path.exists()

    def revision(self) -> int | None:
        """A cheap change token used to decide whether a cached model is stale.

        Reads file metadata only, never the contents.
        """
        try:
            return self.model_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def load(self) -> ModelArtifact:
        """Load and validate the active model.

        Raises:
            ModelNotAvailableError: if no model file is present.
            InvalidModelArtifactError: if the file exists but is unusable.
        """
        if not self.exists():
            raise ModelNotAvailableError(
                "No trained model is available. Train a model first."
            )
        return load_artifact(self.model_path)

    def promote(self, artifact: ModelArtifact) -> Path:
        """Persist a freshly trained artifact and make it the active model.

        The artifact is written to a staging path, read back and validated from
        disk, and only then moved into place. Validating the in-memory object
        would prove the object is sound; reading it back proves the *file* is.
        """
        staging_path = self._model_dir / f"{MODEL_FILENAME}.incoming"

        with _promote_lock:
            dump_artifact(artifact, staging_path)
            try:
                load_artifact(staging_path)
            except Exception:
                staging_path.unlink(missing_ok=True)
                logger.exception(
                    "Rejected model %s: validation failed", artifact.version
                )
                raise

            staging_path.replace(self.model_path)

        logger.info(
            "Promoted model %s (%d samples, accuracy=%s)",
            artifact.version,
            artifact.n_samples,
            artifact.accuracy,
        )
        return self.model_path


class ModelHolder:
    """Serves the active model from memory, reloading when the file changes.

    Loading a pickle on every request would dominate inference time. Comparing
    the file's modification token is a metadata lookup costing microseconds, so
    a retrain becomes visible on the very next prediction without a restart.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._artifact: ModelArtifact | None = None
        self._revision: int | None = None
        self._lock = threading.Lock()

    def get(self) -> ModelArtifact:
        """Return the current model, reloading it if the file has changed."""
        revision = self._registry.revision()
        if revision is None:
            self.invalidate()
            raise ModelNotAvailableError(
                "No trained model is available. Train a model first."
            )

        with self._lock:
            if self._artifact is None or revision != self._revision:
                logger.info("Loading model from disk (revision %s)", revision)
                self._artifact = self._registry.load()
                self._revision = revision
            return self._artifact

    def try_get(self) -> ModelArtifact | None:
        """Return the current model, or ``None`` if none is usable.

        Used by pages that must render whether or not a model exists.
        """
        try:
            return self.get()
        except Exception as exc:  # noqa: BLE001 - the UI only needs "unavailable"
            logger.warning("Model unavailable: %s", exc)
            return None

    def invalidate(self) -> None:
        with self._lock:
            self._artifact = None
            self._revision = None