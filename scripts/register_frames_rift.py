#!/usr/bin/env python3
"""Register extracted infrared/visible frames with RIFT."""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::UserWarning:src.phase_congruency.tools",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"src\.phase_congruency\.tools",
)

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
RIFT_SRC = REPO_ROOT / "RIFT-Multimodal-matching-python"
if str(RIFT_SRC) not in sys.path:
    sys.path.insert(0, str(RIFT_SRC))

from src.RIFT2 import RIFT2  # noqa: E402


DEFAULT_INPUT_ROOT = Path("data/video_data_20260521/frames")
DEFAULT_OUTPUT_ROOT = Path("data/video_data_20260521/rift_registration")
TARGET_SIZE = (640, 512)


@dataclass(frozen=True)
class FramePair:
    frame_id: str
    infrared_path: Path
    visible_path: Path


@dataclass
class RegistrationResult:
    frame_id: str
    status: str
    matches: int = 0
    inliers: int = 0
    message: str = ""
    homography: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register infrared and visible extracted frames with RIFT."
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
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1,
        help="Maximum number of frame pairs to process. Use 0 for all pairs. Default: 1",
    )
    parser.add_argument(
        "--lowes-ratio",
        type=float,
        default=0.95,
        help="Lowe ratio used for descriptor matching. Default: 0.95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-frame logs and RIFT internal messages.",
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


def read_inputs(pair: FramePair) -> tuple[np.ndarray, np.ndarray]:
    infrared = cv2.imread(str(pair.infrared_path), cv2.IMREAD_COLOR)
    visible = cv2.imread(str(pair.visible_path), cv2.IMREAD_COLOR)

    if infrared is None:
        raise RuntimeError(f"Failed to read infrared image: {pair.infrared_path}")
    if visible is None:
        raise RuntimeError(f"Failed to read visible image: {pair.visible_path}")

    if (infrared.shape[1], infrared.shape[0]) != TARGET_SIZE:
        infrared = cv2.resize(infrared, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    if (visible.shape[1], visible.shape[0]) != TARGET_SIZE:
        visible = cv2.resize(visible, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return infrared, visible


def match_descriptors(
    des1: np.ndarray,
    des2: np.ndarray,
    kp1,
    kp2,
    lowes_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[cv2.DMatch]]:
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

    matches_knn = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good_matches = []
    for match_pair in matches_knn:
        if len(match_pair) != 2:
            continue
        m, n = match_pair
        if m.distance < lowes_ratio * n.distance:
            good_matches.append(m)

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    return points1, points2, good_matches


def make_overlay(infrared: np.ndarray, warped_visible: np.ndarray) -> np.ndarray:
    infrared_gray = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    infrared_color = cv2.cvtColor(infrared_gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(infrared_color, 0.5, warped_visible, 0.5, 0)


def make_checkerboard(
    infrared: np.ndarray,
    warped_visible: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    rows, cols = infrared.shape[:2]
    yy, xx = np.indices((rows, cols))
    mask = ((yy // block_size + xx // block_size) % 2).astype(bool)
    checkerboard = infrared.copy()
    checkerboard[mask] = warped_visible[mask]
    return checkerboard


def label_image(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (230, 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def make_preview(
    infrared: np.ndarray,
    warped_visible: np.ndarray,
    overlay: np.ndarray,
    checkerboard: np.ndarray,
) -> np.ndarray:
    top = np.hstack(
        [
            label_image(overlay, "overlay"),
            label_image(checkerboard, "checkerboard"),
        ]
    )
    bottom = np.hstack(
        [
            label_image(warped_visible, "warped visible"),
            label_image(infrared, "infrared"),
        ]
    )
    return np.vstack([top, bottom])


def flatten_homography(homography: np.ndarray | None) -> list[str]:
    if homography is None:
        return [""] * 9
    return [f"{value:.10g}" for value in homography.reshape(-1)]


def parse_existing_results(output_root: Path) -> dict[str, RegistrationResult]:
    csv_path = output_root / "homographies.csv"
    if not csv_path.exists():
        return {}

    results: dict[str, RegistrationResult] = {}
    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_id = row.get("frame_id", "")
            if not frame_id:
                continue

            homography_values = [row.get(f"h{i}{j}", "") for i in range(3) for j in range(3)]
            homography = None
            if all(value != "" for value in homography_values):
                homography = np.array([float(value) for value in homography_values]).reshape(3, 3)

            results[frame_id] = RegistrationResult(
                frame_id=frame_id,
                status=row.get("status", ""),
                matches=int(row.get("matches") or 0),
                inliers=int(row.get("inliers") or 0),
                message=row.get("message", ""),
                homography=homography,
            )
    return results


def register_pair(
    pair: FramePair,
    rift: RIFT2,
    output_root: Path,
    lowes_ratio: float,
    overwrite: bool,
    existing_results: dict[str, RegistrationResult],
    verbose: bool,
) -> RegistrationResult:
    warped_dir = output_root / "warped_rgb"
    preview_dir = output_root / "preview"
    warped_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    warped_path = warped_dir / f"visible_warped_{pair.frame_id}.jpg"
    preview_path = preview_dir / f"preview_{pair.frame_id}.jpg"

    if warped_path.exists() and preview_path.exists() and not overwrite:
        existing_result = existing_results.get(pair.frame_id)
        if existing_result is not None and existing_result.homography is not None:
            return existing_result
        return RegistrationResult(frame_id=pair.frame_id, status="skipped", message="outputs already exist")

    infrared, visible = read_inputs(pair)
    if verbose:
        kp_infrared, des_infrared, kp_visible, des_visible = rift(infrared, visible)
    else:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            kp_infrared, des_infrared, kp_visible, des_visible = rift(infrared, visible)
    points_infrared, points_visible, matches = match_descriptors(
        des_infrared,
        des_visible,
        kp_infrared,
        kp_visible,
        lowes_ratio=lowes_ratio,
    )

    if len(matches) < 4:
        return RegistrationResult(
            frame_id=pair.frame_id,
            status="failed",
            matches=len(matches),
            message="not enough matches",
        )

    homography, mask = cv2.findHomography(
        points_visible,
        points_infrared,
        cv2.USAC_MAGSAC,
        5.0,
    )
    if homography is None or mask is None:
        return RegistrationResult(
            frame_id=pair.frame_id,
            status="failed",
            matches=len(matches),
            message="homography estimation failed",
        )

    inliers = int(mask.ravel().sum())
    warped_visible = cv2.warpPerspective(visible, homography, TARGET_SIZE)
    overlay = make_overlay(infrared, warped_visible)
    checkerboard = make_checkerboard(infrared, warped_visible)
    preview = make_preview(infrared, warped_visible, overlay, checkerboard)

    if not cv2.imwrite(str(warped_path), warped_visible):
        raise RuntimeError(f"Failed to write warped image: {warped_path}")
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to write preview image: {preview_path}")

    return RegistrationResult(
        frame_id=pair.frame_id,
        status="ok",
        matches=len(matches),
        inliers=inliers,
        homography=homography,
    )


def write_results_csv(output_root: Path, results: list[RegistrationResult]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "homographies.csv"
    fieldnames = [
        "frame_id",
        "status",
        "matches",
        "inliers",
        "message",
        "h00",
        "h01",
        "h02",
        "h10",
        "h11",
        "h12",
        "h20",
        "h21",
        "h22",
    ]

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for result in results:
            writer.writerow(
                [
                    result.frame_id,
                    result.status,
                    result.matches,
                    result.inliers,
                    result.message,
                    *flatten_homography(result.homography),
                ]
            )


def main() -> int:
    args = parse_args()
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be >= 0")
    if not 0 < args.lowes_ratio <= 1:
        raise ValueError("--lowes-ratio must be in (0, 1]")

    pairs = collect_frame_pairs(args.input_root, args.max_pairs)
    print(f"found_pairs={len(pairs)} target_size={TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    rift = RIFT2()
    existing_results = parse_existing_results(args.output_root)
    results: list[RegistrationResult] = []
    progress = tqdm(pairs, desc="registering", unit="pair", dynamic_ncols=True)
    for pair in progress:
        progress.set_postfix(frame=pair.frame_id, refresh=False)
        try:
            result = register_pair(
                pair=pair,
                rift=rift,
                output_root=args.output_root,
                lowes_ratio=args.lowes_ratio,
                overwrite=args.overwrite,
                existing_results=existing_results,
                verbose=args.verbose,
            )
        except Exception as exc:
            result = RegistrationResult(
                frame_id=pair.frame_id,
                status="failed",
                message=str(exc),
            )
        results.append(result)
        progress.set_postfix(
            frame=result.frame_id,
            status=result.status,
            inliers=result.inliers,
            refresh=False,
        )
        if args.verbose:
            tqdm.write(
                f"frame={result.frame_id} status={result.status} "
                f"matches={result.matches} inliers={result.inliers} {result.message}"
            )

    write_results_csv(args.output_root, results)
    ok_count = sum(result.status == "ok" for result in results)
    failed_count = sum(result.status == "failed" for result in results)
    skipped_count = sum(result.status == "skipped" for result in results)
    print(
        f"done ok={ok_count} failed={failed_count} skipped={skipped_count} "
        f"output={args.output_root}"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
