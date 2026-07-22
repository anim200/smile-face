"""Import the Kaggle smiling/not-smiling dataset into the project layout.

Usage:
    python scripts/bootstrap_dataset.py path/to/extracted/kaggle/folder

The Kaggle archive uses folder names "smile" and "non_smile". This normalises
them to the project's canonical labels and re-encodes everything to JPEG, so
downstream code only ever sees one format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.core.constants import (  # noqa: E402
    JPEG_QUALITY,
    LABEL_NOT_SMILING,
    LABEL_SMILING,
)

#: Kaggle folder name -> project label. Extra spellings are cheap insurance.
SOURCE_FOLDERS = {
    "smile": LABEL_SMILING,
    "smiling": LABEL_SMILING,
    "non_smile": LABEL_NOT_SMILING,
    "not_smile": LABEL_NOT_SMILING,
    "non_smiling": LABEL_NOT_SMILING,
}
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted Kaggle dataset folder")
    parser.add_argument(
        "--dest", type=Path, default=Path("data/dataset"), help="Target dataset folder"
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"error: {args.source} is not a directory", file=sys.stderr)
        return 1

    totals: dict[str, int] = {}
    for folder in sorted(args.source.rglob("*")):
        label = SOURCE_FOLDERS.get(folder.name.lower())
        if not folder.is_dir() or label is None:
            continue

        target = args.dest / label
        target.mkdir(parents=True, exist_ok=True)
        copied = 0

        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.convert("RGB").save(
                        target / f"{path.stem}.jpg", "JPEG", quality=JPEG_QUALITY
                    )
                copied += 1
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"  skipped {path.name}: {exc}", file=sys.stderr)

        totals[label] = totals.get(label, 0) + copied
        print(f"{folder.name} -> {label}: {copied} images")

    if not totals:
        print("error: no recognised class folders found", file=sys.stderr)
        print(f"       expected one of: {sorted(SOURCE_FOLDERS)}", file=sys.stderr)
        return 1

    print(f"\nDataset ready at {args.dest}: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())