#!/usr/bin/env python3
"""Create direct infrared/visible overlay images before registration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


DEFAULT_INPUT_ROOT = Path("data/video_data_20260521/frames_undistorted")
DEFAULT_OUTPUT_DIR = Path("data/video_data_20260521/rift_registration/raw_overlay")
TARGET_SIZE = (640, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create direct overlays from undistorted infrared/visible frames."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root containing infrared/ and visible/ folders. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for raw overlay images. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Visible image weight in overlay. Default: 0.5",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum frame pairs to process. Use 0 for all pairs. Default: 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing overlay images.",
    )
    return parser.parse_args()


def frame_id_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def collect_pairs(input_root: Path, max_pairs: int) -> list[tuple[str, Path, Path]]:
    infrared_dir = input_root / "infrared"
    visible_dir = input_root / "visible"
    if not infrared_dir.is_dir():
        raise FileNotFoundError(f"Missing infrared directory: {infrared_dir}")
    if not visible_dir.is_dir():
        raise FileNotFoundError(f"Missing visible directory: {visible_dir}")

    infrared_by_id = {
        frame_id_from_path(path): path
        for path in sorted(infrared_dir.glob("infrared_*.jpg"))
    }
    visible_by_id = {
        frame_id_from_path(path): path
        for path in sorted(visible_dir.glob("visible_*.jpg"))
    }
    common_ids = sorted(set(infrared_by_id) & set(visible_by_id))
    if not common_ids:
        raise RuntimeError(f"No matching infrared/visible frame pairs found in {input_root}")
    if max_pairs > 0:
        common_ids = common_ids[:max_pairs]
    return [(frame_id, infrared_by_id[frame_id], visible_by_id[frame_id]) for frame_id in common_ids]


def read_resized(path: Path) -> cv2.typing.MatLike:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if (image.shape[1], image.shape[0]) != TARGET_SIZE:
        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return image


def make_overlay(infrared: cv2.typing.MatLike, visible: cv2.typing.MatLike, alpha: float) -> cv2.typing.MatLike:
    infrared_gray = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    infrared_color = cv2.cvtColor(infrared_gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(infrared_color, 1.0 - alpha, visible, alpha, 0)


def main() -> int:
    args = parse_args()
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be >= 0")
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    pairs = collect_pairs(args.input_root, args.max_pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"found_pairs={len(pairs)} output={args.output_dir}")

    written = 0
    skipped = 0
    with tqdm(pairs, desc="raw overlay", unit="pair", dynamic_ncols=True) as progress:
        for frame_id, infrared_path, visible_path in progress:
            output_path = args.output_dir / f"raw_overlay_{frame_id}.jpg"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                progress.set_postfix(frame=frame_id, status="skipped", refresh=False)
                continue

            infrared = read_resized(infrared_path)
            visible = read_resized(visible_path)
            overlay = make_overlay(infrared, visible, args.alpha)
            if not cv2.imwrite(str(output_path), overlay):
                raise RuntimeError(f"Failed to write overlay image: {output_path}")
            written += 1
            progress.set_postfix(frame=frame_id, status="written", refresh=False)

    print(f"done written={written} skipped={skipped} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
