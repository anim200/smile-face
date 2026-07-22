"""Database engine and session management."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.base import Base

logger = logging.getLogger(__name__)

_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    # Verifies a pooled connection before handing it out. Without this, a
    # connection dropped by a database restart surfaces as a failed request.
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session that always closes."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def init_database() -> None:
    """Create any missing tables.

    Importing the models module here is required, not decorative: SQLAlchemy
    registers a table on ``Base.metadata`` when the model class is defined, so
    ``create_all`` sees nothing unless the module has been imported first.

    Adequate for a project of this size. A longer lived system would use Alembic
    migrations, since ``create_all`` cannot alter an existing table.
    """
    from app.db import models  # noqa: F401  - registers the ORM mappings

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")