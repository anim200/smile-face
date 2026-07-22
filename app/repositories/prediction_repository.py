"""Data access for prediction records.

Routers talk to repositories rather than writing queries inline, so the query
logic is testable on its own and the web layer stays free of SQL.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Prediction


class PredictionRepository:
    """Reads and writes ``predictions`` rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        image_path: str,
        image_url: str,
        original_filename: str,
        predicted_class: str,
        confidence: float | None,
        model_version: str | None,
    ) -> Prediction:
        """Persist one classification result and return it with its id."""
        prediction = Prediction(
            image_path=image_path,
            image_url=image_url,
            original_filename=original_filename,
            predicted_class=predicted_class,
            confidence=confidence,
            model_version=model_version,
        )
        self._session.add(prediction)
        self._session.commit()
        self._session.refresh(prediction)
        return prediction

    def get(self, prediction_id: int) -> Prediction | None:
        return self._session.get(Prediction, prediction_id)

    def list_recent(self, *, limit: int = 20, offset: int = 0) -> list[Prediction]:
        """Newest first, which is the order the History page displays."""
        statement = (
            select(Prediction)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self._session.scalars(statement))

    def count(self) -> int:
        return self._session.scalar(select(func.count()).select_from(Prediction)) or 0

    def count_by_class(self) -> dict[str, int]:
        """Class totals, used for the summary shown above the history table."""
        statement = select(Prediction.predicted_class, func.count()).group_by(
            Prediction.predicted_class
        )
        return {label: total for label, total in self._session.execute(statement)}