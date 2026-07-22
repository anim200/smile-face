"""Data access for training run records."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import STATUS_FAILED, STATUS_RUNNING, STATUS_SUCCESS, TrainingRun


class TrainingRunRepository:
    """Reads and writes ``training_runs`` rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def start(self, *, smiling: int, not_smiling: int) -> TrainingRun:
        """Record the beginning of a run, before any fitting happens."""
        run = TrainingRun(
            status=STATUS_RUNNING,
            images_used=smiling + not_smiling,
            smiling_count=smiling,
            not_smiling_count=not_smiling,
        )
        self._session.add(run)
        self._session.commit()
        self._session.refresh(run)
        return run

    def mark_success(
        self,
        run_id: int,
        *,
        accuracy: float | None,
        f1_smiling: float | None,
        model_version: str,
    ) -> None:
        run = self._session.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = STATUS_SUCCESS
        run.accuracy = accuracy
        run.f1_smiling = f1_smiling
        run.model_version = model_version
        run.finished_at = datetime.now(timezone.utc)
        self._session.commit()

    def mark_failed(self, run_id: int, *, error: str) -> None:
        """Record a failure. The previously active model remains in service."""
        run = self._session.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = STATUS_FAILED
        run.error = error[:2000]
        run.finished_at = datetime.now(timezone.utc)
        self._session.commit()

    def has_active_run(self) -> bool:
        """True if a run is in progress, used to reject concurrent training."""
        statement = select(TrainingRun.id).where(TrainingRun.status == STATUS_RUNNING)
        return self._session.scalars(statement).first() is not None

    def list_recent(self, *, limit: int = 10) -> list[TrainingRun]:
        statement = select(TrainingRun).order_by(TrainingRun.id.desc()).limit(limit)
        return list(self._session.scalars(statement))