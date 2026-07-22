"""Orchestrates a training run.

Runs as a background task after the response has been sent, so the browser is
never left waiting on a model fit. The old model stays active until a new one
has been trained, written, and validated: a failed run is a non-event.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.constants import LABEL_NOT_SMILING, LABEL_SMILING
from app.db.session import SessionLocal
from app.ml.registry import ModelRegistry
from app.ml.train import train_from_directory
from app.repositories.training_run_repository import TrainingRunRepository
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


def run_training(*, run_id: int, staging_dir: Path, model_dir: Path) -> None:
    """Fit a model from the staged images and promote it if it is sound.

    A background task outlives the request that scheduled it, so this opens its
    own session rather than reusing the one the router was given.
    """
    session = SessionLocal()
    runs = TrainingRunRepository(session)
    dataset = DatasetService(staging_dir)

    try:
        counts = dataset.counts()
        logger.info("Training run %s starting on %s", run_id, counts)

        artifact = train_from_directory(
            staging_dir,
            source=(
                f"Uploaded via Train page: {counts[LABEL_SMILING]} smiling, "
                f"{counts[LABEL_NOT_SMILING]} not smiling"
            ),
        )

        # Promote validates the written file before replacing the active model.
        ModelRegistry(model_dir).promote(artifact)

        runs.mark_success(
            run_id,
            accuracy=artifact.accuracy,
            f1_smiling=artifact.metrics.get("f1_smiling"),
            model_version=artifact.version,
        )

        # Requirement: the uploaded images are removed once training is done.
        dataset.clear()
        logger.info("Training run %s finished as %s", run_id, artifact.version)

    except Exception as exc:  # noqa: BLE001 - every failure is recorded, not raised
        logger.exception("Training run %s failed", run_id)
        runs.mark_failed(run_id, error=f"{type(exc).__name__}: {exc}")
        # Staged files are deliberately left in place so the user can retry
        # without uploading everything again.
    finally:
        session.close()