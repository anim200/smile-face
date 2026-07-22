"""Application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import get_settings
from app.core.constants import CLASS_DISPLAY_NAMES
from app.core.logging import configure_logging
from app.db.session import init_database
from app.ml.registry import ModelHolder, ModelRegistry
from app.routers import classify, history, home, train

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Prepare shared state before the first request is served."""
    configure_logging(settings.log_level)
    settings.ensure_directories()
    init_database()

    holder = ModelHolder(ModelRegistry(settings.model_dir))
    app.state.model_holder = holder

    # Attempt an eager load so a broken model is reported at startup rather
    # than on a user's first classification. Absence is not an error: the app
    # must boot so that a model can be trained through it.
    artifact = holder.try_get()
    if artifact is None:
        logger.warning("Starting without an active model")
    else:
        logger.info(
            "Active model %s (accuracy %s)", artifact.version, artifact.accuracy
        )

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name, lifespan=lifespan, docs_url=None, redoc_url=None
    )

    templates = Jinja2Templates(directory=str(settings.base_dir / "app" / "templates"))
    templates.env.globals["app_name"] = settings.app_name
    templates.env.globals["class_names"] = CLASS_DISPLAY_NAMES
    app.state.templates = templates

    app.mount(
        "/static",
        StaticFiles(directory=str(settings.base_dir / "app" / "static")),
        name="static",
    )
    # Saved prediction images are served from the data volume, not the package.
    settings.predictions_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/media",
        StaticFiles(directory=str(settings.predictions_dir)),
        name="media",
    )

    # Order matters only for readability; the four menu items of the brief.
    app.include_router(home.router)
    app.include_router(train.router)
    app.include_router(classify.router)
    app.include_router(history.router)

    @app.exception_handler(404)
    async def not_found(request: Request, exc) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "error.html", {"message": "Page not found"}, status_code=404
        )

    return app


app = create_app()