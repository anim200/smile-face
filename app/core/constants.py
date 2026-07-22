"""Domain constants shared across the application.

Kept free of imports from other application modules so that any layer can
depend on it without creating a cycle.
"""

from typing import Final

LABEL_SMILING: Final[str] = "smiling"
LABEL_NOT_SMILING: Final[str] = "not_smiling"

#: Canonical class order. Stable and safe to persist.
CLASS_LABELS: Final[tuple[str, ...]] = (LABEL_NOT_SMILING, LABEL_SMILING)

#: Human readable names for the UI layer.
CLASS_DISPLAY_NAMES: Final[dict[str, str]] = {
    LABEL_NOT_SMILING: "Not smiling",
    LABEL_SMILING: "Smiling",
}

#: Upload content types accepted before Pillow re-encoding.
ALLOWED_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/bmp"}
)

JPEG_QUALITY: Final[int] = 90