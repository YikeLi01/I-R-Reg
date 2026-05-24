#!/usr/bin/env python3
"""Run video frame extraction, undistortion, and RIFT registration."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path


DEFAULT_VIDEO_DIR = Path("data/video_data_20260521")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extraction, undistortion, and RIFT registration for infrared/visible videos."
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
        help="Output directory for registration. Default: <video-dir>/rift_registration or rift2_registration",
    )
    parser.add_argument(
        "--undistorted-frames-dir",
        type=Path,
        default=None,
        help="Output directory for undistorted frames. Default: <video-dir>/frames_undistorted",
    )
    parser.add_argument(
        "--ir-config",
        type=Path,
        default=Path("config/ir_mono.yaml"),
        help="IR mono calibration YAML. Default: config/ir_mono.yaml",
    )
    parser.add_argument(
        "--rgb-config",
        type=Path,
        default=Path("config/rgb_mono.yaml"),
        help="RGB mono calibration YAML. Default: config/rgb_mono.yaml",
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
        "--min-inliers",
        type=int,
        default=50,
        help="Minimum homography inliers required to accept registration. Default: 50",
    )
    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.05,
        help="Minimum inliers/matches ratio required to accept registration. Default: 0.05",
    )
    parser.add_argument(
        "--max-projective",
        type=float,
        default=1e-3,
        help="Maximum absolute h20/h21 projective terms for accepted H. Default: 1e-3",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.5,
        help="Minimum affine scale allowed for accepted H. Default: 0.5",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=2.0,
        help="Maximum affine scale allowed for accepted H. Default: 2.0",
    )
    parser.add_argument(
        "--max-translation",
        type=float,
        default=640.0,
        help="Maximum absolute x/y translation allowed for accepted H. Default: 640",
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
        "--register-backend",
        choices=("rift", "rift2"),
        default="rift",
        help="Registration backend to run. Default: rift",
    )
    parser.add_argument(
        "--no-fallback-h",
        action="store_true",
        help="Disable using the last accepted homography when registration quality gates fail.",
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
    process = subprocess.Popen(command, cwd=cwd)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print(f"\ninterrupting {label} ...", file=sys.stderr, flush=True)
        try:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"{label} did not stop after SIGINT; terminating", file=sys.stderr, flush=True)
            process.terminate()
            return_code = process.wait(timeout=10)
        if return_code != 0:
            raise KeyboardInterrupt
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


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


def build_undistort_command(
    args: argparse.Namespace,
    frames_dir: Path,
    undistorted_frames_dir: Path,
    root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "scripts" / "undistort_extracted_frames.py"),
        "--input-root",
        str(frames_dir),
        "--output-root",
        str(undistorted_frames_dir),
        "--ir-config",
        str(args.ir_config),
        "--rgb-config",
        str(args.rgb_config),
        "--max-pairs",
        str(args.max_pairs),
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
    register_script = (
        "register_frames_rift2.py"
        if args.register_backend == "rift2"
        else "register_frames_rift.py"
    )
    command = [
        sys.executable,
        str(root / "scripts" / register_script),
        "--input-root",
        str(frames_dir),
        "--output-root",
        str(registration_dir),
        "--max-pairs",
        str(args.max_pairs),
        "--lowes-ratio",
        str(args.lowes_ratio),
        "--min-inliers",
        str(args.min_inliers),
        "--min-inlier-ratio",
        str(args.min_inlier_ratio),
        "--max-projective",
        str(args.max_projective),
        "--min-scale",
        str(args.min_scale),
        "--max-scale",
        str(args.max_scale),
        "--max-translation",
        str(args.max_translation),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.no_fallback_h:
        command.append("--no-fallback-h")
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
    if args.min_inliers < 4:
        raise ValueError("--min-inliers must be >= 4")
    if not 0 <= args.min_inlier_ratio <= 1:
        raise ValueError("--min-inlier-ratio must be between 0 and 1")
    if args.max_projective <= 0:
        raise ValueError("--max-projective must be greater than 0")
    if args.min_scale <= 0 or args.max_scale < args.min_scale:
        raise ValueError("--min-scale/--max-scale values are invalid")
    if args.max_translation <= 0:
        raise ValueError("--max-translation must be greater than 0")

    root = repo_root()
    args.video_dir = resolve_path(args.video_dir_pos or args.video_dir, root)
    frames_dir = resolve_path(args.frames_dir, root) if args.frames_dir else args.video_dir / "frames"
    undistorted_frames_dir = (
        resolve_path(args.undistorted_frames_dir, root)
        if args.undistorted_frames_dir
        else args.video_dir / "frames_undistorted"
    )
    registration_dir = (
        resolve_path(args.registration_dir, root)
        if args.registration_dir
        else args.video_dir / (
            "rift2_registration" if args.register_backend == "rift2" else "rift_registration"
        )
    )
    args.ir_config = resolve_path(args.ir_config, root)
    args.rgb_config = resolve_path(args.rgb_config, root)

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
    if not args.ir_config.is_file():
        raise FileNotFoundError(f"Missing IR calibration file: {args.ir_config}")
    if not args.rgb_config.is_file():
        raise FileNotFoundError(f"Missing RGB calibration file: {args.rgb_config}")

    print("pipeline configuration", flush=True)
    print(f"video_dir={args.video_dir}", flush=True)
    print(f"infrared_video={args.infrared_video.name}", flush=True)
    print(f"visible_video={args.visible_video.name}", flush=True)
    print(f"frames_dir={frames_dir}", flush=True)
    print(f"undistorted_frames_dir={undistorted_frames_dir}", flush=True)
    print(f"register_backend={args.register_backend}", flush=True)
    print(f"registration_dir={registration_dir}", flush=True)
    print(f"ir_config={args.ir_config}", flush=True)
    print(f"rgb_config={args.rgb_config}", flush=True)
    print(f"step={args.step} max_pairs={args.max_pairs} overwrite={args.overwrite}", flush=True)

    try:
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

        if not has_extracted_frames(frames_dir):
            raise RuntimeError(f"No extracted frame pairs found in {frames_dir}")
        if has_extracted_frames(undistorted_frames_dir) and not args.overwrite:
            print("\n== undistort frames ==\nskipped because undistorted frames already exist", flush=True)
        else:
            run_command(
                build_undistort_command(args, frames_dir, undistorted_frames_dir, root),
                cwd=root,
                label="undistort frames",
            )

        if args.skip_register:
            print("\n== registration ==\nskipped by --skip-register", flush=True)
        else:
            if not has_extracted_frames(undistorted_frames_dir):
                raise RuntimeError(f"No undistorted frame pairs found in {undistorted_frames_dir}")
            run_command(
                build_register_command(args, undistorted_frames_dir, registration_dir, root),
                cwd=root,
                label=f"{args.register_backend} registration",
            )
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr, flush=True)
        return 130
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 130:
            print("\ninterrupted by user", file=sys.stderr, flush=True)
            return 130
        raise

    print("\npipeline done", flush=True)
    print(f"frames: {frames_dir}", flush=True)
    print(f"undistorted frames: {undistorted_frames_dir}", flush=True)
    print(f"registration: {registration_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
