"""Train a smile classifier from the command line and save it as a pickle file.

Usage:
    python scripts/train.py
    python scripts/train.py --data path --models path

Calls exactly the same training function the web application uses, so a model
produced here is indistinguishable from one produced by the Train page.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging  # noqa: E402
from app.ml.registry import ModelRegistry  # noqa: E402
from app.ml.train import count_images_per_class, train_from_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING" if args.quiet else "INFO")

    counts = count_images_per_class(args.data)
    if not any(counts.values()):
        print(f"error: no images under {args.data}", file=sys.stderr)
        print("       run scripts/bootstrap_dataset.py first", file=sys.stderr)
        return 1
    print(f"Dataset: {counts}")

    started = time.perf_counter()
    artifact = train_from_directory(args.data, source=f"CLI: {args.data}")
    elapsed = time.perf_counter() - started

    args.models.mkdir(parents=True, exist_ok=True)
    path = ModelRegistry(args.models).promote(artifact)

    metrics = artifact.metrics
    print(f"\nTrained in {elapsed:.1f}s using {metrics['strategy']}")
    print(f"  accuracy    {metrics['accuracy']:.4f}")
    print(f"  f1 smiling  {metrics['f1_smiling']:.4f}")
    print(f"  confusion   {metrics['confusion_matrix']}  order={metrics['label_order']}")
    print(f"\nSaved {path} (version {artifact.version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())