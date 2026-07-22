"""Train page: stage labelled uploads, then fit a new model from them."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.config import get_settings
from app.core.constants import CLASS_LABELS
from app.core.exceptions import InvalidImageError
from app.dependencies import ModelHolderDep, TemplatesDep, TrainingRepoDep
from app.services.dataset_service import DatasetService
from app.services.image_service import save_upload_as_jpeg
from app.services.training_service import run_training

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/train")
settings = get_settings()


def _page_context(holder, runs, *, message=None, error=None) -> dict:
    dataset = DatasetService(settings.staging_dir)
    return {
        "active_page": "train",
        "model": holder.try_get(),
        "counts": dataset.counts(),
        "ready": dataset.is_ready(settings.min_images_per_class),
        "minimum": settings.min_images_per_class,
        "runs": runs.list_recent(limit=5),
        "busy": runs.has_active_run(),
        "labels": CLASS_LABELS,
        "message": message,
        "error": error,
    }


@router.get("", response_class=HTMLResponse)
async def train_page(
    request: Request,
    templates: TemplatesDep,
    holder: ModelHolderDep,
    runs: TrainingRepoDep,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "train.html", _page_context(holder, runs)
    )


@router.post("/upload")
async def upload_images(
    request: Request,
    templates: TemplatesDep,
    holder: ModelHolderDep,
    runs: TrainingRepoDep,
    label: str = Form(...),
    files: list[UploadFile] = File(...),
) -> HTMLResponse:
    """Stage a batch of images under one class label."""
    if label not in CLASS_LABELS:
        return templates.TemplateResponse(
            request,
            "train.html",
            _page_context(holder, runs, error=f"Unknown label {label!r}"),
            status_code=400,
        )

    if len(files) > settings.max_files_per_upload:
        return templates.TemplateResponse(
            request,
            "train.html",
            _page_context(
                holder,
                runs,
                error=f"At most {settings.max_files_per_upload} files at a time",
            ),
            status_code=400,
        )

    dataset = DatasetService(settings.staging_dir)
    destination = dataset.directory_for(label)

    saved, rejected = 0, []
    for upload in files:
        try:
            await save_upload_as_jpeg(
                upload, destination, max_bytes=settings.max_upload_bytes
            )
            saved += 1
        except InvalidImageError as exc:
            rejected.append(str(exc))

    message = f"Added {saved} image{'' if saved == 1 else 's'} to {label.replace('_', ' ')}."
    error = "; ".join(rejected[:3]) if rejected else None
    logger.info("Staged %d image(s) as %s, %d rejected", saved, label, len(rejected))

    return templates.TemplateResponse(
        request, "train.html", _page_context(holder, runs, message=message, error=error)
    )


@router.post("/start")
async def start_training(
    request: Request,
    templates: TemplatesDep,
    holder: ModelHolderDep,
    runs: TrainingRepoDep,
    background: BackgroundTasks,
):
    """Schedule a training run over the staged images."""
    dataset = DatasetService(settings.staging_dir)

    if runs.has_active_run():
        return templates.TemplateResponse(
            request,
            "train.html",
            _page_context(holder, runs, error="A training run is already in progress."),
            status_code=409,
        )

    if not dataset.is_ready(settings.min_images_per_class):
        return templates.TemplateResponse(
            request,
            "train.html",
            _page_context(
                holder,
                runs,
                error=(
                    f"Both classes need at least {settings.min_images_per_class} "
                    "images before training can start."
                ),
            ),
            status_code=400,
        )

    counts = dataset.counts()
    run = runs.start(smiling=counts["smiling"], not_smiling=counts["not_smiling"])

    background.add_task(
        run_training,
        run_id=run.id,
        staging_dir=settings.staging_dir,
        model_dir=settings.model_dir,
    )
    return RedirectResponse(url="/train?started=1", status_code=303)


@router.post("/clear")
async def clear_staging(
    request: Request,
    templates: TemplatesDep,
    holder: ModelHolderDep,
    runs: TrainingRepoDep,
) -> HTMLResponse:
    DatasetService(settings.staging_dir).clear()
    return templates.TemplateResponse(
        request, "train.html", _page_context(holder, runs, message="Staged images cleared.")
    )