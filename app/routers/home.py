"""Home page: explains the approach and reports the active model."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.dependencies import ModelHolderDep, TemplatesDep

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(
    request: Request, templates: TemplatesDep, holder: ModelHolderDep
) -> HTMLResponse:
    artifact = holder.try_get()
    return templates.TemplateResponse(
        request,
        "home.html",
        {"active_page": "home", "model": artifact},
    )