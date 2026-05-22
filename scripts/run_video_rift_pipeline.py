#!/usr/bin/env python3
"""Run video frame extraction followed by RIFT registration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_VIDEO_DIR = Path("data/video_data_20260521")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction and RIFT registration for infrared/visible videos."
    )
    parser.add_argument(
        "video_dir_pos",
        nargs="?",
        type=Path,
        help="Directory containing source videos. Overrides --video-dir when provided.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory containing source videos. Default: {DEFAULT_VIDEO_DIR}",
    )
    parser.add_argument(
        "--infrared-name",
        default=None,
        help="Infrared video filename. Default: auto-detect *infrared* video in video directory.",
    )
    parser.add_argument(
        "--visible-name",
        default=None,
        help="Visible video filename. Default: auto-detect *visible* video in video directory.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Output directory for extracted frames. Default: <video-dir>/frames",
    )
    parser.add_argument(
        "--registration-dir",
        type=Path,
        default=None,
        help="Output directory for RIFT registration. Default: <video-dir>/rift_registration",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Save one image every N video frames. Default: 10",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality for extracted frames. Default: 95",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum frame pairs to register. Use 0 for all pairs. Default: 0",
    )
    parser.add_argument(
        "--lowes-ratio",
        type=float,
        default=0.95,
        help="Lowe ratio used by RIFT descriptor matching. Default: 0.95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite extracted frames and registration outputs.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip video frame extraction.",
    )
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Skip RIFT registration.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass verbose logging to the registration stage.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def has_extracted_frames(frames_dir: Path) -> bool:
    infrared_dir = frames_dir / "infrared"
    visible_dir = frames_dir / "visible"
    return (
        infrared_dir.is_dir()
        and visible_dir.is_dir()
        and any(infrared_dir.glob("infrared_*.jpg"))
        and any(visible_dir.glob("visible_*.jpg"))
    )


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def find_video_by_keyword(video_dir: Path, keyword: str) -> Path:
    candidates = [
        path
        for path in sorted(video_dir.iterdir())
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and keyword.lower() in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No *{keyword}* video found in {video_dir}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(
            f"Multiple *{keyword}* videos found in {video_dir}: {names}. "
            f"Specify --{keyword}-name explicitly."
        )
    return candidates[0]


def run_command(command: list[str], cwd: Path, label: str) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def build_extract_command(args: argparse.Namespace, frames_dir: Path, root: Path) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts" / "extract_video_frames.py"),
        "--input-dir",
        str(args.video_dir),
        "--infrared-name",
        args.infrared_video.name,
        "--visible-name",
        args.visible_video.name,
        "--output-dir",
        str(frames_dir),
        "--step",
        str(args.step),
        "--jpg-quality",
        str(args.jpg_quality),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def build_register_command(
    args: argparse.Namespace,
    frames_dir: Path,
    registration_dir: Path,
    root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts" / "register_frames_rift.py"),
        "--input-root",
        str(frames_dir),
        "--output-root",
        str(registration_dir),
        "--max-pairs",
        str(args.max_pairs),
        "--lowes-ratio",
        str(args.lowes_ratio),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.verbose:
        command.append("--verbose")
    return command


def main() -> int:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be greater than 0")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be between 1 and 100")
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be >= 0")
    if not 0 < args.lowes_ratio <= 1:
        raise ValueError("--lowes-ratio must be in (0, 1]")

    root = repo_root()
    args.video_dir = resolve_path(args.video_dir_pos or args.video_dir, root)
    frames_dir = resolve_path(args.frames_dir, root) if args.frames_dir else args.video_dir / "frames"
    registration_dir = (
        resolve_path(args.registration_dir, root)
        if args.registration_dir
        else args.video_dir / "rift_registration"
    )

    args.infrared_video = (
        args.video_dir / args.infrared_name
        if args.infrared_name
        else find_video_by_keyword(args.video_dir, "infrared")
    )
    args.visible_video = (
        args.video_dir / args.visible_name
        if args.visible_name
        else find_video_by_keyword(args.video_dir, "visible")
    )
    if not args.skip_extract:
        if not args.infrared_video.is_file():
            raise FileNotFoundError(f"Missing infrared video: {args.infrared_video}")
        if not args.visible_video.is_file():
            raise FileNotFoundError(f"Missing visible video: {args.visible_video}")

    print("pipeline configuration", flush=True)
    print(f"video_dir={args.video_dir}", flush=True)
    print(f"infrared_video={args.infrared_video.name}", flush=True)
    print(f"visible_video={args.visible_video.name}", flush=True)
    print(f"frames_dir={frames_dir}", flush=True)
    print(f"registration_dir={registration_dir}", flush=True)
    print(f"step={args.step} max_pairs={args.max_pairs} overwrite={args.overwrite}", flush=True)

    if args.skip_extract:
        print("\n== extract frames ==\nskipped by --skip-extract", flush=True)
    elif has_extracted_frames(frames_dir) and not args.overwrite:
        print("\n== extract frames ==\nskipped because extracted frames already exist", flush=True)
    else:
        run_command(
            build_extract_command(args, frames_dir, root),
            cwd=root,
            label="extract frames",
        )

    if args.skip_register:
        print("\n== RIFT registration ==\nskipped by --skip-register", flush=True)
    else:
        if not has_extracted_frames(frames_dir):
            raise RuntimeError(f"No extracted frame pairs found in {frames_dir}")
        run_command(
            build_register_command(args, frames_dir, registration_dir, root),
            cwd=root,
            label="RIFT registration",
        )

    print("\npipeline done", flush=True)
    print(f"frames: {frames_dir}", flush=True)
    print(f"registration: {registration_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
