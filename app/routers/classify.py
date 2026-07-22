"""Classify page: upload one image, store the result, show it on its own page."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.config import get_settings
from app.core.exceptions import InvalidImageError, SmileClassifierError
from app.dependencies import (
    ModelHolderDep,
    PredictionRepoDep,
    PredictorDep,
    TemplatesDep,
)
from app.services.image_service import (
    dated_subdirectory,
    relative_media_url,
    save_upload_as_jpeg,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classify")
settings = get_settings()


@router.get("", response_class=HTMLResponse)
async def classify_form(
    request: Request, templates: TemplatesDep, holder: ModelHolderDep
) -> HTMLResponse:
    """Show the upload form, or an explanation if no model is available."""
    return templates.TemplateResponse(
        request,
        "classify.html",
        {"active_page": "classify", "model": holder.try_get(), "error": None},
    )


@router.post("")
async def classify_submit(
    request: Request,
    templates: TemplatesDep,
    holder: ModelHolderDep,
    predictor: PredictorDep,
    repository: PredictionRepoDep,
    file: UploadFile = File(...),
):
    """Store the upload, classify it, persist the result, then redirect."""
    try:
        image_path = await save_upload_as_jpeg(
            file,
            dated_subdirectory(settings.predictions_dir),
            max_bytes=settings.max_upload_bytes,
        )
    except InvalidImageError as exc:
        return templates.TemplateResponse(
            request,
            "classify.html",
            {"active_page": "classify", "model": holder.try_get(), "error": str(exc)},
            status_code=400,
        )

    try:
        result = predictor.predict(image_path)
    except SmileClassifierError as exc:
        image_path.unlink(missing_ok=True)
        return templates.TemplateResponse(
            request,
            "classify.html",
            {"active_page": "classify", "model": holder.try_get(), "error": str(exc)},
            status_code=503,
        )

    prediction = repository.create(
        image_path=str(image_path),
        image_url=relative_media_url(image_path, settings.predictions_dir),
        original_filename=file.filename or "upload.jpg",
        predicted_class=result.label,
        confidence=result.confidence,
        model_version=result.model_version,
    )
    logger.info(
        "Classified %s as %s (%.3f)",
        prediction.original_filename,
        result.label,
        result.confidence,
    )

    # Post/Redirect/Get: refreshing the result page must not resubmit the file.
    return RedirectResponse(url=f"/classify/{prediction.id}", status_code=303)


@router.get("/{prediction_id}", response_class=HTMLResponse)
async def classify_result(
    prediction_id: int,
    request: Request,
    templates: TemplatesDep,
    repository: PredictionRepoDep,
) -> HTMLResponse:
    prediction = repository.get(prediction_id)
    if prediction is None:
        return templates.TemplateResponse(
            request,
            "error.html",
            {"active_page": "classify", "message": "That result no longer exists"},
            status_code=404,
        )
    return templates.TemplateResponse(
        request,
        "result.html",
        {"active_page": "classify", "prediction": prediction},
    )