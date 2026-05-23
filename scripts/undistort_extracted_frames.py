#!/usr/bin/env python3
"""Undistort extracted infrared and visible frames using mono calibration files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


DEFAULT_INPUT_ROOT = Path("data/video_data_20260521/frames")
DEFAULT_IR_CONFIG = Path("config/ir_mono.yaml")
DEFAULT_RGB_CONFIG = Path("config/rgb_mono.yaml")


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray


@dataclass(frozen=True)
class FramePair:
    frame_id: str
    infrared_path: Path
    visible_path: Path


@dataclass
class UndistortStats:
    processed_pairs: int = 0
    saved_infrared: int = 0
    saved_visible: int = 0
    skipped_infrared: int = 0
    skipped_visible: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undistort extracted infrared and visible frames."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Root containing infrared/ and visible/ frame folders. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root directory. Default: <input-root parent>/frames_undistorted",
    )
    parser.add_argument(
        "--ir-config",
        type=Path,
        default=DEFAULT_IR_CONFIG,
        help=f"IR mono calibration YAML. Default: {DEFAULT_IR_CONFIG}",
    )
    parser.add_argument(
        "--rgb-config",
        type=Path,
        default=DEFAULT_RGB_CONFIG,
        help=f"RGB mono calibration YAML. Default: {DEFAULT_RGB_CONFIG}",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum frame pairs to process. Use 0 for all pairs. Default: 0",
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
        help="Overwrite existing undistorted images.",
    )
    return parser.parse_args()


def frame_id_from_path(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def collect_frame_pairs(input_root: Path, max_pairs: int) -> list[FramePair]:
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

    return [
        FramePair(
            frame_id=frame_id,
            infrared_path=infrared_by_id[frame_id],
            visible_path=visible_by_id[frame_id],
        )
        for frame_id in common_ids
    ]


def load_calibration(config_path: Path) -> CameraCalibration:
    with config_path.open("r") as file:
        data = yaml.safe_load(file)

    distortion_model = data.get("distortion_model")
    if distortion_model != "opencv_pinhole_k1_k2_p1_p2_k3":
        raise ValueError(
            f"Unsupported distortion_model in {config_path}: {distortion_model}"
        )

    image_size = tuple(int(value) for value in data["image_size"])
    if len(image_size) != 2:
        raise ValueError(f"Invalid image_size in {config_path}: {image_size}")

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3 in {config_path}")
    if dist_coeffs.size < 4:
        raise ValueError(f"dist_coeffs must contain at least 4 values in {config_path}")

    map_x, map_y = cv2.initUndistortRectifyMap(
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        R=None,
        newCameraMatrix=camera_matrix,
        size=image_size,
        m1type=cv2.CV_32FC1,
    )

    return CameraCalibration(
        name=str(data.get("camera_name", config_path.stem)),
        image_size=image_size,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        map_x=map_x,
        map_y=map_y,
    )


def read_image(path: Path, calibration: CameraCalibration) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    expected_width, expected_height = calibration.image_size
    actual_height, actual_width = image.shape[:2]
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ValueError(
            f"Image size mismatch for {path}: "
            f"got {actual_width}x{actual_height}, "
            f"expected {expected_width}x{expected_height} from {calibration.name} calibration"
        )

    return image


def undistort_image(image: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    return cv2.remap(
        image,
        calibration.map_x,
        calibration.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def write_image(path: Path, image: np.ndarray, jpg_quality: int, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
    )
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")
    return True


def process_pair(
    pair: FramePair,
    output_root: Path,
    ir_calibration: CameraCalibration,
    rgb_calibration: CameraCalibration,
    jpg_quality: int,
    overwrite: bool,
) -> tuple[bool, bool]:
    infrared_output = output_root / "infrared" / pair.infrared_path.name
    visible_output = output_root / "visible" / pair.visible_path.name

    infrared_saved = False
    if overwrite or not infrared_output.exists():
        infrared = read_image(pair.infrared_path, ir_calibration)
        undistorted_infrared = undistort_image(infrared, ir_calibration)
        infrared_saved = write_image(
            infrared_output,
            undistorted_infrared,
            jpg_quality=jpg_quality,
            overwrite=overwrite,
        )

    visible_saved = False
    if overwrite or not visible_output.exists():
        visible = read_image(pair.visible_path, rgb_calibration)
        undistorted_visible = undistort_image(visible, rgb_calibration)
        visible_saved = write_image(
            visible_output,
            undistorted_visible,
            jpg_quality=jpg_quality,
            overwrite=overwrite,
        )

    return infrared_saved, visible_saved


def main() -> int:
    args = parse_args()
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be >= 0")
    if not 1 <= args.jpg_quality <= 100:
        raise ValueError("--jpg-quality must be between 1 and 100")

    output_root = args.output_root or (args.input_root.parent / "frames_undistorted")
    pairs = collect_frame_pairs(args.input_root, args.max_pairs)
    ir_calibration = load_calibration(args.ir_config)
    rgb_calibration = load_calibration(args.rgb_config)

    print(f"found_pairs={len(pairs)}")
    print(f"input_root={args.input_root}")
    print(f"output_root={output_root}")
    print(f"ir_size={ir_calibration.image_size[0]}x{ir_calibration.image_size[1]}")
    print(f"rgb_size={rgb_calibration.image_size[0]}x{rgb_calibration.image_size[1]}")

    stats = UndistortStats()
    with tqdm(pairs, desc="undistorting", unit="pair", dynamic_ncols=True) as progress:
        for pair in progress:
            infrared_saved, visible_saved = process_pair(
                pair=pair,
                output_root=output_root,
                ir_calibration=ir_calibration,
                rgb_calibration=rgb_calibration,
                jpg_quality=args.jpg_quality,
                overwrite=args.overwrite,
            )
            stats.processed_pairs += 1
            if infrared_saved:
                stats.saved_infrared += 1
            else:
                stats.skipped_infrared += 1
            if visible_saved:
                stats.saved_visible += 1
            else:
                stats.skipped_visible += 1

            progress.set_postfix(frame=pair.frame_id, refresh=False)

    print(
        "done "
        f"pairs={stats.processed_pairs} "
        f"infrared_saved={stats.saved_infrared} "
        f"infrared_skipped={stats.skipped_infrared} "
        f"visible_saved={stats.saved_visible} "
        f"visible_skipped={stats.skipped_visible}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
