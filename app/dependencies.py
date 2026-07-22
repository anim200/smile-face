"""FastAPI dependency providers.

Shared objects are created once at startup and handed to routers through these
functions, so routers never construct their own database sessions or model
loaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.ml.infer import SmilePredictor
from app.ml.registry import ModelHolder
from app.repositories.prediction_repository import PredictionRepository
from app.repositories.training_run_repository import TrainingRunRepository

SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[Session, Depends(get_db)]


def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


def get_model_holder(request: Request) -> ModelHolder:
    """The single shared model holder created at startup."""
    return request.app.state.model_holder


def get_predictor(
    holder: Annotated[ModelHolder, Depends(get_model_holder)],
) -> SmilePredictor:
    return SmilePredictor(holder)


def get_prediction_repository(session: SessionDep) -> PredictionRepository:
    return PredictionRepository(session)


def get_training_run_repository(session: SessionDep) -> TrainingRunRepository:
    return TrainingRunRepository(session)


TemplatesDep = Annotated[Jinja2Templates, Depends(get_templates)]
ModelHolderDep = Annotated[ModelHolder, Depends(get_model_holder)]
PredictorDep = Annotated[SmilePredictor, Depends(get_predictor)]
PredictionRepoDep = Annotated[
    PredictionRepository, Depends(get_prediction_repository)
]
TrainingRepoDep = Annotated[
    TrainingRunRepository, Depends(get_training_run_repository)
]


def resolve_path(settings: Settings, *parts: str) -> Path:
    """Join paths relative to the configured base directory."""
    return settings.base_dir.joinpath(*parts)