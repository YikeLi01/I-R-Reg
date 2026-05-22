#!/usr/bin/env python3
"""Extract synchronized frames from infrared and visible videos."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2


DEFAULT_INPUT_DIR = Path("data/video_data_20260521")
DEFAULT_INFRARED_NAME = "infrared_20260521_204552.mp4"
DEFAULT_VISIBLE_NAME = "visible_20260521_204552.mp4"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "frames"


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class ExtractStats:
    processed_frames: int
    saved_infrared: int = 0
    saved_visible: int = 0
    skipped_infrared: int = 0
    skipped_visible: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract aligned images from infrared and visible videos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing the input videos. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--infrared-name",
        default=DEFAULT_INFRARED_NAME,
        help=f"Infrared video filename. Default: {DEFAULT_INFRARED_NAME}",
    )
    parser.add_argument(
        "--visible-name",
        default=DEFAULT_VISIBLE_NAME,
        help=f"Visible video filename. Default: {DEFAULT_VISIBLE_NAME}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Save one image every N frames. Default: 10",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100. Default: 95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted images.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.step <= 0:
        raise ValueError("--step must be greater than 0")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be between 1 and 100")


def read_video_meta(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"Could not determine frame count for video: {path}")

    return VideoMeta(
        path=path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )


def print_meta(label: str, meta: VideoMeta) -> None:
    print(
        f"{label}: path={meta.path}, frames={meta.frame_count}, "
        f"fps={meta.fps:g}, size={meta.width}x{meta.height}"
    )


def save_frame(
    image,
    output_path: Path,
    jpg_quality: int,
    overwrite: bool,
) -> bool:
    if output_path.exists() and not overwrite:
        return False

    ok = cv2.imwrite(
        str(output_path),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
    )
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")
    return True


def extract_frames(
    infrared_path: Path,
    visible_path: Path,
    output_dir: Path,
    step: int,
    jpg_quality: int,
    overwrite: bool,
) -> ExtractStats:
    infrared_meta = read_video_meta(infrared_path)
    visible_meta = read_video_meta(visible_path)

    print_meta("infrared", infrared_meta)
    print_meta("visible", visible_meta)

    if not math.isclose(infrared_meta.fps, visible_meta.fps, rel_tol=0.0, abs_tol=0.01):
        print(
            "warning: input videos have different FPS values; "
            f"infrared={infrared_meta.fps:g}, visible={visible_meta.fps:g}"
        )

    min_frame_count = min(infrared_meta.frame_count, visible_meta.frame_count)
    if infrared_meta.frame_count != visible_meta.frame_count:
        print(
            "warning: input videos have different frame counts; "
            f"using the shorter length: {min_frame_count}"
        )

    infrared_output_dir = output_dir / "infrared"
    visible_output_dir = output_dir / "visible"
    infrared_output_dir.mkdir(parents=True, exist_ok=True)
    visible_output_dir.mkdir(parents=True, exist_ok=True)

    infrared_cap = cv2.VideoCapture(str(infrared_path))
    visible_cap = cv2.VideoCapture(str(visible_path))
    stats = ExtractStats(processed_frames=min_frame_count)

    try:
        if not infrared_cap.isOpened():
            raise RuntimeError(f"Could not open video: {infrared_path}")
        if not visible_cap.isOpened():
            raise RuntimeError(f"Could not open video: {visible_path}")

        for frame_index in range(min_frame_count):
            infrared_ok, infrared_frame = infrared_cap.read()
            visible_ok, visible_frame = visible_cap.read()

            if not infrared_ok:
                raise RuntimeError(f"Failed to read infrared frame {frame_index}")
            if not visible_ok:
                raise RuntimeError(f"Failed to read visible frame {frame_index}")

            if frame_index % step != 0:
                continue

            infrared_output = infrared_output_dir / f"infrared_{frame_index:06d}.jpg"
            visible_output = visible_output_dir / f"visible_{frame_index:06d}.jpg"

            if save_frame(infrared_frame, infrared_output, jpg_quality, overwrite):
                stats.saved_infrared += 1
            else:
                stats.skipped_infrared += 1

            if save_frame(visible_frame, visible_output, jpg_quality, overwrite):
                stats.saved_visible += 1
            else:
                stats.skipped_visible += 1
    finally:
        infrared_cap.release()
        visible_cap.release()

    return stats


def main() -> int:
    args = parse_args()
    validate_args(args)

    infrared_path = args.input_dir / args.infrared_name
    visible_path = args.input_dir / args.visible_name

    stats = extract_frames(
        infrared_path=infrared_path,
        visible_path=visible_path,
        output_dir=args.output_dir,
        step=args.step,
        jpg_quality=args.jpg_quality,
        overwrite=args.overwrite,
    )

    expected = (stats.processed_frames + args.step - 1) // args.step
    print("done")
    print(f"processed_frames={stats.processed_frames}")
    print(f"expected_images_per_video={expected}")
    print(
        f"infrared: saved={stats.saved_infrared}, skipped={stats.skipped_infrared}, "
        f"output={args.output_dir / 'infrared'}"
    )
    print(
        f"visible: saved={stats.saved_visible}, skipped={stats.skipped_visible}, "
        f"output={args.output_dir / 'visible'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
