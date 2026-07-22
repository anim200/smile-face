"""Classify images from the command line using the saved pickle file.

Usage:
    python scripts/predict.py photo.jpg [more.jpg ...]

Loading the model in a separate process from training is the check that matters:
if this works, the web application will load the same file without trouble.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.constants import CLASS_DISPLAY_NAMES  # noqa: E402
from app.core.exceptions import InvalidImageError, SmileClassifierError  # noqa: E402
from app.ml.infer import SmilePredictor  # noqa: E402
from app.ml.registry import ModelHolder, ModelRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()

    holder = ModelHolder(ModelRegistry(args.models))
    try:
        artifact = holder.get()
    except SmileClassifierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Model v{artifact.version}, trained {artifact.trained_at:%Y-%m-%d %H:%M} UTC")
    print(f"Accuracy {artifact.accuracy}, {artifact.n_samples} training images\n")

    predictor = SmilePredictor(holder)
    failures = 0
    for image_path in args.images:
        try:
            result = predictor.predict(image_path)
        except InvalidImageError as exc:
            print(f"{image_path.name:<30} error: {exc}", file=sys.stderr)
            failures += 1
            continue
        label = CLASS_DISPLAY_NAMES[result.label]
        print(f"{image_path.name:<30} {label:<12} {result.confidence:.1%}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())