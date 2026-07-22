"""Image loading and feature extraction.

HOG is implemented here in numpy rather than pulled from scikit-image. It is a
single well-defined algorithm, and implementing it directly removes a large
compiled dependency for no measurable loss in accuracy.

``HOGTransformer`` is deliberately defined in this module rather than in a
script. Pickle stores only the import path of a class, so a transformer defined
in ``__main__`` cannot be unpickled by the web process. Keeping it here means
the training script and the API resolve it to the same symbol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.base import BaseEstimator, TransformerMixin

from app.core.exceptions import InvalidImageError

DEFAULT_IMAGE_SIZE = 64
_EPS = 1e-7
#: Contrast ceiling applied before renormalising, the "Hys" in L2-Hys.
_CLIP = 0.2


def load_grayscale(path: Path | str, size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Load one image as a normalised ``(size, size)`` float32 array.

    Raises:
        InvalidImageError: if the file cannot be decoded as an image.
    """
    try:
        with Image.open(path) as image:
            resized = image.convert("L").resize((size, size), Image.BILINEAR)
            array = np.asarray(resized, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Cannot read image: {path}") from exc
    return array / 255.0


def load_grayscale_batch(
    paths: Iterable[Path | str], size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Load many images into a single ``(n, size, size)`` array."""
    images = [load_grayscale(path, size) for path in paths]
    if not images:
        return np.empty((0, size, size), dtype=np.float32)
    return np.stack(images)


def hog_descriptor(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Compute a Histogram of Oriented Gradients descriptor.

    The image is divided into cells; each pixel votes into an orientation
    histogram for its cell, weighted by gradient magnitude. Overlapping blocks
    of cells are then normalised together, which is what makes the descriptor
    robust to lighting changes.
    """
    img = np.asarray(image, dtype=np.float64)
    height, width = img.shape

    # Central differences. Borders stay zero, which suppresses frame artefacts.
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    magnitude = np.hypot(gx, gy)
    # Unsigned gradients: a dark-to-light edge and its reverse are equivalent.
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180.0

    cell_h, cell_w = pixels_per_cell
    n_cells_y, n_cells_x = height // cell_h, width // cell_w
    bin_width = 180.0 / orientations

    magnitude = _to_cells(magnitude, n_cells_y, n_cells_x, cell_h, cell_w)
    orientation = _to_cells(orientation, n_cells_y, n_cells_x, cell_h, cell_w)

    # Split each vote between the two nearest bins so small rotations produce
    # small changes in the descriptor rather than a discontinuous jump.
    position = orientation / bin_width - 0.5
    lower = np.floor(position).astype(int)
    fraction = position - lower

    histogram = np.zeros((n_cells_y, n_cells_x, orientations))
    rows, cols = np.meshgrid(np.arange(n_cells_y), np.arange(n_cells_x), indexing="ij")
    rows, cols = rows[..., None], cols[..., None]
    np.add.at(histogram, (rows, cols, lower % orientations), magnitude * (1 - fraction))
    np.add.at(histogram, (rows, cols, (lower + 1) % orientations), magnitude * fraction)

    return _normalise_blocks(histogram, cells_per_block)


def _to_cells(
    values: np.ndarray, n_cells_y: int, n_cells_x: int, cell_h: int, cell_w: int
) -> np.ndarray:
    """Reshape a pixel grid into ``(n_cells_y, n_cells_x, pixels_per_cell)``."""
    cropped = values[: n_cells_y * cell_h, : n_cells_x * cell_w]
    return (
        cropped.reshape(n_cells_y, cell_h, n_cells_x, cell_w)
        .transpose(0, 2, 1, 3)
        .reshape(n_cells_y, n_cells_x, -1)
    )


def _normalise_blocks(
    histogram: np.ndarray, cells_per_block: tuple[int, int]
) -> np.ndarray:
    """Apply L2-Hys normalisation over overlapping blocks and flatten."""
    block_h, block_w = cells_per_block
    n_cells_y, n_cells_x, orientations = histogram.shape
    n_blocks_y = n_cells_y - block_h + 1
    n_blocks_x = n_cells_x - block_w + 1

    blocks = np.zeros((n_blocks_y, n_blocks_x, block_h, block_w, orientations))
    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            block = histogram[y : y + block_h, x : x + block_w, :]
            block = block / np.sqrt(np.sum(block**2) + _EPS**2)
            block = np.clip(block, 0, _CLIP)
            blocks[y, x] = block / np.sqrt(np.sum(block**2) + _EPS**2)

    return blocks.ravel()


class HOGTransformer(BaseEstimator, TransformerMixin):
    """Turn grayscale images into HOG feature vectors.

    HOG encodes the direction of local intensity changes, which is what a mouth
    corner and a nasolabial fold actually are. It is far more robust to lighting
    variation than raw pixel values, and cheap enough to run on a CPU.
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
    ) -> None:
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X: Sequence[np.ndarray], y=None) -> "HOGTransformer":  # noqa: N803
        return self

    def transform(self, X: Sequence[np.ndarray]) -> np.ndarray:  # noqa: N803
        return np.asarray(
            [
                hog_descriptor(
                    image,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                )
                for image in X
            ],
            dtype=np.float32,
        )