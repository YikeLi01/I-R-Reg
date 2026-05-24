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
MAX_INLIER_LINES = 100
IMAGE_CORNERS = np.array(
    [
        [0.0, 0.0, 1.0],
        [TARGET_SIZE[0] - 1.0, 0.0, 1.0],
        [TARGET_SIZE[0] - 1.0, TARGET_SIZE[1] - 1.0, 1.0],
        [0.0, TARGET_SIZE[1] - 1.0, 1.0],
    ],
    dtype=np.float64,
).T


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
        "--transform-model",
        choices=("homography", "affine"),
        default="homography",
        help="Geometric model to estimate from RIFT matches. Default: homography",
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
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--no-fallback-h",
        action="store_true",
        help="Disable using the last accepted homography when quality gates fail.",
    )
    parser.add_argument(
        "--no-temporal-gate",
        action="store_true",
        help="Disable checking the current homography against the last usable homography.",
    )
    parser.add_argument(
        "--max-corner-mean-shift",
        type=float,
        default=80.0,
        help="Maximum mean corner displacement from the last usable H. Default: 80",
    )
    parser.add_argument(
        "--max-corner-max-shift",
        type=float,
        default=160.0,
        help="Maximum single-corner displacement from the last usable H. Default: 160",
    )
    parser.add_argument(
        "--max-temporal-translation",
        type=float,
        default=80.0,
        help="Maximum x/y translation change from the last usable H. Default: 80",
    )
    parser.add_argument(
        "--min-temporal-scale-ratio",
        type=float,
        default=0.75,
        help="Minimum scale ratio against the last usable H. Default: 0.75",
    )
    parser.add_argument(
        "--max-temporal-scale-ratio",
        type=float,
        default=1.33,
        help="Maximum scale ratio against the last usable H. Default: 1.33",
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


def write_registration_images(
    warped_path: Path,
    preview_path: Path,
    infrared: np.ndarray,
    visible_output: np.ndarray,
) -> None:
    overlay = make_overlay(infrared, visible_output)
    checkerboard = make_checkerboard(infrared, visible_output)
    preview = make_preview(infrared, visible_output, overlay, checkerboard)

    if not cv2.imwrite(str(warped_path), visible_output):
        raise RuntimeError(f"Failed to write warped image: {warped_path}")
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Failed to write preview image: {preview_path}")


def make_inlier_matches_image(
    infrared: np.ndarray,
    visible: np.ndarray,
    kp_infrared,
    kp_visible,
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray | None,
    status: str,
    message: str,
) -> np.ndarray:
    canvas = np.hstack([infrared.copy(), visible.copy()])
    height, width = infrared.shape[:2]

    title = f"status={status} inliers=0 matches={len(matches)}"
    if message:
        title = f"{title} {message}"

    if inlier_mask is not None:
        mask_values = inlier_mask.ravel().astype(bool)
        inlier_matches = [
            match
            for match, is_inlier in zip(matches, mask_values)
            if is_inlier
        ]
        inlier_matches = sorted(inlier_matches, key=lambda match: match.distance)
        inlier_matches = inlier_matches[:MAX_INLIER_LINES]
        title = (
            f"status={status} shown={len(inlier_matches)} "
            f"inliers={int(mask_values.sum())} matches={len(matches)}"
        )
        if message:
            title = f"{title} {message}"

        for index, match in enumerate(inlier_matches):
            x1, y1 = kp_infrared[match.queryIdx].pt
            x2, y2 = kp_visible[match.trainIdx].pt
            point1 = (int(round(x1)), int(round(y1)))
            point2 = (int(round(x2)) + width, int(round(y2)))
            color = (
                int(37 + (index * 53) % 190),
                int(220 - (index * 47) % 160),
                int(60 + (index * 31) % 180),
            )
            cv2.line(canvas, point1, point2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, point1, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, point2, 3, color, -1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        title[:150],
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (width, 0), (width, height), (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def write_inlier_matches_image(
    inlier_matches_path: Path,
    infrared: np.ndarray,
    visible: np.ndarray,
    kp_infrared,
    kp_visible,
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray | None,
    status: str,
    message: str,
) -> None:
    image = make_inlier_matches_image(
        infrared=infrared,
        visible=visible,
        kp_infrared=kp_infrared,
        kp_visible=kp_visible,
        matches=matches,
        inlier_mask=inlier_mask,
        status=status,
        message=message,
    )
    if not cv2.imwrite(str(inlier_matches_path), image):
        raise RuntimeError(f"Failed to write inlier matches image: {inlier_matches_path}")


def write_failed_or_fallback_images(
    warped_path: Path,
    preview_path: Path,
    infrared: np.ndarray,
    visible: np.ndarray,
    fallback_homography: np.ndarray | None,
) -> np.ndarray | None:
    if fallback_homography is None:
        write_registration_images(warped_path, preview_path, infrared, visible)
        return None

    warped_visible = cv2.warpPerspective(visible, fallback_homography, TARGET_SIZE)
    write_registration_images(warped_path, preview_path, infrared, warped_visible)
    return fallback_homography


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


def validate_homography(
    homography: np.ndarray,
    matches: int,
    inliers: int,
    min_inliers: int,
    min_inlier_ratio: float,
    max_projective: float,
    min_scale: float,
    max_scale: float,
    max_translation: float,
    check_projective: bool = True,
) -> str | None:
    if inliers < min_inliers:
        return f"low_inliers:{inliers}<{min_inliers}"

    inlier_ratio = inliers / max(matches, 1)
    if inlier_ratio < min_inlier_ratio:
        return f"low_inlier_ratio:{inlier_ratio:.4f}<{min_inlier_ratio:.4f}"

    if check_projective and (
        abs(float(homography[2, 0])) > max_projective
        or abs(float(homography[2, 1])) > max_projective
    ):
        return (
            f"large_projective_terms:"
            f"h20={homography[2, 0]:.6g},h21={homography[2, 1]:.6g}"
        )

    scale_x = float(np.linalg.norm(homography[0:2, 0]))
    scale_y = float(np.linalg.norm(homography[0:2, 1]))
    if not (min_scale <= scale_x <= max_scale and min_scale <= scale_y <= max_scale):
        return f"scale_out_of_range:sx={scale_x:.4f},sy={scale_y:.4f}"

    tx = float(homography[0, 2])
    ty = float(homography[1, 2])
    if abs(tx) > max_translation or abs(ty) > max_translation:
        return f"translation_out_of_range:tx={tx:.2f},ty={ty:.2f}"

    return None


def estimate_transform(
    points_visible: np.ndarray,
    points_infrared: np.ndarray,
    transform_model: str,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if transform_model == "homography":
        homography, mask = cv2.findHomography(
            points_visible,
            points_infrared,
            cv2.USAC_MAGSAC,
            5.0,
        )
        return homography, mask, "homography estimation failed"

    affine, mask = cv2.estimateAffine2D(
        points_visible,
        points_infrared,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )
    if affine is None:
        return None, mask, "affine estimation failed"

    homography = np.eye(3, dtype=np.float64)
    homography[:2, :] = affine
    return homography, mask, "affine estimation failed"


def homography_scale(homography: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(homography[0:2, 0])),
        float(np.linalg.norm(homography[0:2, 1])),
    )


def project_image_corners(homography: np.ndarray) -> np.ndarray:
    projected = homography @ IMAGE_CORNERS
    return (projected[:2] / projected[2]).T


def validate_temporal_homography(
    homography: np.ndarray,
    reference_homography: np.ndarray | None,
    max_corner_mean_shift: float,
    max_corner_max_shift: float,
    max_temporal_translation: float,
    min_temporal_scale_ratio: float,
    max_temporal_scale_ratio: float,
) -> str | None:
    if reference_homography is None:
        return None

    current_corners = project_image_corners(homography)
    reference_corners = project_image_corners(reference_homography)
    corner_shifts = np.linalg.norm(current_corners - reference_corners, axis=1)
    mean_shift = float(corner_shifts.mean())
    max_shift = float(corner_shifts.max())
    if mean_shift > max_corner_mean_shift:
        return f"temporal_jump:corner_mean_shift={mean_shift:.2f}>{max_corner_mean_shift:.2f}"
    if max_shift > max_corner_max_shift:
        return f"temporal_jump:corner_max_shift={max_shift:.2f}>{max_corner_max_shift:.2f}"

    tx_delta = abs(float(homography[0, 2] - reference_homography[0, 2]))
    ty_delta = abs(float(homography[1, 2] - reference_homography[1, 2]))
    if tx_delta > max_temporal_translation or ty_delta > max_temporal_translation:
        return (
            f"temporal_jump:translation_delta="
            f"tx={tx_delta:.2f},ty={ty_delta:.2f}>{max_temporal_translation:.2f}"
        )

    scale_x, scale_y = homography_scale(homography)
    reference_scale_x, reference_scale_y = homography_scale(reference_homography)
    scale_ratio_x = scale_x / max(reference_scale_x, 1e-12)
    scale_ratio_y = scale_y / max(reference_scale_y, 1e-12)
    if not (
        min_temporal_scale_ratio <= scale_ratio_x <= max_temporal_scale_ratio
        and min_temporal_scale_ratio <= scale_ratio_y <= max_temporal_scale_ratio
    ):
        return (
            f"temporal_jump:scale_ratio="
            f"sx={scale_ratio_x:.4f},sy={scale_ratio_y:.4f}"
            f" not_in [{min_temporal_scale_ratio:.4f},{max_temporal_scale_ratio:.4f}]"
        )

    return None


def register_pair(
    pair: FramePair,
    rift: RIFT2,
    output_root: Path,
    lowes_ratio: float,
    overwrite: bool,
    existing_results: dict[str, RegistrationResult],
    verbose: bool,
    transform_model: str,
    min_inliers: int,
    min_inlier_ratio: float,
    max_projective: float,
    min_scale: float,
    max_scale: float,
    max_translation: float,
    fallback_homography: np.ndarray | None = None,
    reference_homography: np.ndarray | None = None,
    max_corner_mean_shift: float = 80.0,
    max_corner_max_shift: float = 160.0,
    max_temporal_translation: float = 80.0,
    min_temporal_scale_ratio: float = 0.75,
    max_temporal_scale_ratio: float = 1.33,
) -> RegistrationResult:
    warped_dir = output_root / "warped_rgb"
    preview_dir = output_root / "preview"
    inlier_matches_dir = output_root / "inlier_matches"
    warped_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    inlier_matches_dir.mkdir(parents=True, exist_ok=True)

    warped_path = warped_dir / f"visible_warped_{pair.frame_id}.jpg"
    preview_path = preview_dir / f"preview_{pair.frame_id}.jpg"
    inlier_matches_path = inlier_matches_dir / f"inliers_{pair.frame_id}.jpg"

    if (
        warped_path.exists()
        and preview_path.exists()
        and inlier_matches_path.exists()
        and not overwrite
    ):
        existing_result = existing_results.get(pair.frame_id)
        if (
            existing_result is not None
            and existing_result.status in {"ok", "fallback"}
            and existing_result.homography is not None
        ):
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
        used_homography = write_failed_or_fallback_images(
            warped_path,
            preview_path,
            infrared,
            visible,
            fallback_homography,
        )
        status = "fallback" if used_homography is not None else "failed"
        message = "not enough matches"
        write_inlier_matches_image(
            inlier_matches_path,
            infrared,
            visible,
            kp_infrared,
            kp_visible,
            matches,
            None,
            status,
            message,
        )
        return RegistrationResult(
            frame_id=pair.frame_id,
            status=status,
            matches=len(matches),
            message=message,
            homography=used_homography,
        )

    homography, mask, estimation_failure_message = estimate_transform(
        points_visible=points_visible,
        points_infrared=points_infrared,
        transform_model=transform_model,
    )
    if homography is None or mask is None:
        used_homography = write_failed_or_fallback_images(
            warped_path,
            preview_path,
            infrared,
            visible,
            fallback_homography,
        )
        status = "fallback" if used_homography is not None else "failed"
        message = estimation_failure_message
        write_inlier_matches_image(
            inlier_matches_path,
            infrared,
            visible,
            kp_infrared,
            kp_visible,
            matches,
            None,
            status,
            message,
        )
        return RegistrationResult(
            frame_id=pair.frame_id,
            status=status,
            matches=len(matches),
            message=message,
            homography=used_homography,
        )

    inliers = int(mask.ravel().sum())
    rejection_reason = validate_homography(
        homography=homography,
        matches=len(matches),
        inliers=inliers,
        min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio,
        max_projective=max_projective,
        min_scale=min_scale,
        max_scale=max_scale,
        max_translation=max_translation,
        check_projective=transform_model == "homography",
    )
    if rejection_reason is not None:
        used_homography = write_failed_or_fallback_images(
            warped_path,
            preview_path,
            infrared,
            visible,
            fallback_homography,
        )
        status = "fallback" if used_homography is not None else "failed"
        write_inlier_matches_image(
            inlier_matches_path,
            infrared,
            visible,
            kp_infrared,
            kp_visible,
            matches,
            mask,
            status,
            rejection_reason,
        )
        return RegistrationResult(
            frame_id=pair.frame_id,
            status=status,
            matches=len(matches),
            inliers=inliers,
            message=rejection_reason,
            homography=used_homography,
        )

    temporal_rejection_reason = validate_temporal_homography(
        homography=homography,
        reference_homography=reference_homography,
        max_corner_mean_shift=max_corner_mean_shift,
        max_corner_max_shift=max_corner_max_shift,
        max_temporal_translation=max_temporal_translation,
        min_temporal_scale_ratio=min_temporal_scale_ratio,
        max_temporal_scale_ratio=max_temporal_scale_ratio,
    )
    if temporal_rejection_reason is not None:
        used_homography = write_failed_or_fallback_images(
            warped_path,
            preview_path,
            infrared,
            visible,
            fallback_homography,
        )
        status = "fallback" if used_homography is not None else "failed"
        write_inlier_matches_image(
            inlier_matches_path,
            infrared,
            visible,
            kp_infrared,
            kp_visible,
            matches,
            mask,
            status,
            temporal_rejection_reason,
        )
        return RegistrationResult(
            frame_id=pair.frame_id,
            status=status,
            matches=len(matches),
            inliers=inliers,
            message=temporal_rejection_reason,
            homography=used_homography,
        )

    warped_visible = cv2.warpPerspective(visible, homography, TARGET_SIZE)
    write_registration_images(warped_path, preview_path, infrared, warped_visible)
    write_inlier_matches_image(
        inlier_matches_path,
        infrared,
        visible,
        kp_infrared,
        kp_visible,
        matches,
        mask,
        "ok",
        "",
    )

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
    tmp_path = output_root / "homographies.csv.tmp"
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

    with tmp_path.open("w", newline="") as file:
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
    os.replace(tmp_path, csv_path)


def sorted_results(results_by_id: dict[str, RegistrationResult]) -> list[RegistrationResult]:
    return [results_by_id[frame_id] for frame_id in sorted(results_by_id)]


def last_usable_homography_before(
    results_by_id: dict[str, RegistrationResult],
    frame_id: str,
) -> np.ndarray | None:
    for existing_frame_id in sorted(results_by_id, reverse=True):
        if existing_frame_id >= frame_id:
            continue
        result = results_by_id[existing_frame_id]
        if result.status in {"ok", "fallback"} and result.homography is not None:
            return result.homography
    return None


def update_last_usable_homography(
    current: np.ndarray | None,
    result: RegistrationResult,
) -> np.ndarray | None:
    if result.status in {"ok", "fallback"} and result.homography is not None:
        return result.homography
    return current


def main() -> int:
    args = parse_args()
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
    if args.max_corner_mean_shift <= 0:
        raise ValueError("--max-corner-mean-shift must be greater than 0")
    if args.max_corner_max_shift <= 0:
        raise ValueError("--max-corner-max-shift must be greater than 0")
    if args.max_temporal_translation <= 0:
        raise ValueError("--max-temporal-translation must be greater than 0")
    if (
        args.min_temporal_scale_ratio <= 0
        or args.max_temporal_scale_ratio < args.min_temporal_scale_ratio
    ):
        raise ValueError("--min-temporal-scale-ratio/--max-temporal-scale-ratio values are invalid")

    pairs = collect_frame_pairs(args.input_root, args.max_pairs)
    print(f"found_pairs={len(pairs)} target_size={TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    rift = RIFT2()
    existing_results = parse_existing_results(args.output_root)
    results_by_id = dict(existing_results)
    processed_results: list[RegistrationResult] = []
    use_fallback_h = not args.no_fallback_h
    use_temporal_gate = not args.no_temporal_gate
    first_frame_id = pairs[0].frame_id
    last_usable_homography = (
        last_usable_homography_before(results_by_id, first_frame_id)
        if use_fallback_h or use_temporal_gate
        else None
    )

    try:
        with tqdm(pairs, desc="registering", unit="pair", dynamic_ncols=True) as progress:
            for pair in progress:
                progress.set_postfix(frame=pair.frame_id, refresh=False)
                try:
                    result = register_pair(
                        pair=pair,
                        rift=rift,
                        output_root=args.output_root,
                        lowes_ratio=args.lowes_ratio,
                        overwrite=args.overwrite,
                        existing_results=results_by_id,
                        verbose=args.verbose,
                        transform_model=args.transform_model,
                        min_inliers=args.min_inliers,
                        min_inlier_ratio=args.min_inlier_ratio,
                        max_projective=args.max_projective,
                        min_scale=args.min_scale,
                        max_scale=args.max_scale,
                        max_translation=args.max_translation,
                        fallback_homography=last_usable_homography if use_fallback_h else None,
                        reference_homography=last_usable_homography if use_temporal_gate else None,
                        max_corner_mean_shift=args.max_corner_mean_shift,
                        max_corner_max_shift=args.max_corner_max_shift,
                        max_temporal_translation=args.max_temporal_translation,
                        min_temporal_scale_ratio=args.min_temporal_scale_ratio,
                        max_temporal_scale_ratio=args.max_temporal_scale_ratio,
                    )
                except Exception as exc:
                    result = RegistrationResult(
                        frame_id=pair.frame_id,
                        status="failed",
                        message=str(exc),
                    )

                results_by_id[result.frame_id] = result
                processed_results.append(result)
                if use_fallback_h or use_temporal_gate:
                    last_usable_homography = update_last_usable_homography(
                        last_usable_homography,
                        result,
                    )
                write_results_csv(args.output_root, sorted_results(results_by_id))

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
    except KeyboardInterrupt:
        write_results_csv(args.output_root, sorted_results(results_by_id))
        print(
            f"\ninterrupted, saved {len(processed_results)} processed result(s) "
            f"to {args.output_root / 'homographies.csv'}",
            file=sys.stderr,
        )
        return 130

    ok_count = sum(result.status == "ok" for result in processed_results)
    fallback_count = sum(result.status == "fallback" for result in processed_results)
    failed_count = sum(result.status == "failed" for result in processed_results)
    skipped_count = sum(result.status == "skipped" for result in processed_results)
    print(
        f"done ok={ok_count} fallback={fallback_count} failed={failed_count} "
        f"skipped={skipped_count} "
        f"output={args.output_root}"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
