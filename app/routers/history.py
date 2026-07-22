"""History page: every stored classification, newest first."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.core.config import get_settings
from app.dependencies import PredictionRepoDep, TemplatesDep

router = APIRouter(prefix="/history")
settings = get_settings()


@router.get("", response_class=HTMLResponse)
async def history(
    request: Request,
    templates: TemplatesDep,
    repository: PredictionRepoDep,
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    """Requirement 14: image, class and date-time for every classification."""
    size = settings.history_page_size
    total = repository.count()
    predictions = repository.list_recent(limit=size, offset=(page - 1) * size)

    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "active_page": "history",
            "predictions": predictions,
            "totals": repository.count_by_class(),
            "total": total,
            # Named page_number/page_count rather than page/pages: Starlette
            # injects its own values into the template context, and a bare
            # "pages" collides with one of them.
            "page_number": page,
            "page_count": max(1, -(-total // size)),
        },
    )