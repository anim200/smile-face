"""Upload handling: validate, convert to JPEG, write to disk.

Re-encoding through Pillow rather than saving the raw bytes is deliberate. It
satisfies the requirement that every image is stored as JPEG, and it doubles as
validation: a file that is not really an image fails to decode and never
reaches the filesystem.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.constants import ALLOWED_CONTENT_TYPES, JPEG_QUALITY
from app.core.exceptions import InvalidImageError

logger = logging.getLogger(__name__)


async def save_upload_as_jpeg(
    upload: UploadFile, destination_dir: Path, *, max_bytes: int
) -> Path:
    """Convert one upload to JPEG and store it under a generated name.

    Returns:
        The path of the written file.

    Raises:
        InvalidImageError: if the type is not allowed, the file is too large,
            or the bytes cannot be decoded as an image.
    """
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(
            f"{upload.filename or 'file'}: unsupported type {upload.content_type}"
        )

    payload = await upload.read()
    if len(payload) > max_bytes:
        raise InvalidImageError(
            f"{upload.filename or 'file'}: larger than {max_bytes // (1024 * 1024)} MB"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / f"{uuid.uuid4().hex}.jpg"

    try:
        with Image.open(BytesIO(payload)) as image:
            image.convert("RGB").save(target, "JPEG", quality=JPEG_QUALITY)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        target.unlink(missing_ok=True)
        raise InvalidImageError(
            f"{upload.filename or 'file'}: not a readable image"
        ) from exc

    return target


def dated_subdirectory(root: Path) -> Path:
    """Return ``root/YYYY/MM``, so one directory never accumulates every file."""
    now = datetime.now(timezone.utc)
    return root / f"{now:%Y}" / f"{now:%m}"


def relative_media_url(path: Path, media_root: Path) -> str:
    """Build the ``/media`` URL for a stored prediction image."""
    return "/media/" + path.relative_to(media_root).as_posix()