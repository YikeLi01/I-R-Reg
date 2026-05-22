# Repository Guidelines

## Project Structure & Module Organization

This repository contains a thermal/visible image-registration codebase. Most source code lives in `TWMM/`:

- `TWMM/common/`: shared image I/O, preprocessing, homography, CSV, and visualization helpers.
- `TWMM/template_based/`: CFOG template-matching implementation.
- `TWMM/feature_based/`: SIFT, SURF, ORB, RIFT, SCB, HardNet, TFeat, and MATLAB-assisted methods.
- `TWMM/TAMM_clean/`: TAMM implementation and a standalone runner.
- `TWMM/test_img/`: sample thermal/visible image pairs and generated result folders.
- `data/video_data_20260521/`: raw video assets.

Keep generated registration outputs under `TWMM/test_img/out*/` or another clearly named output directory.

## Build, Test, and Development Commands

There is no package build system or dependency lock file in this snapshot. Run commands from `TWMM/` unless noted.

- `python results_SIFT_SURF_RIFT_SCB_HOPC.py`: runs the configured registration demo and writes homography, mosaic, match-point, and warp outputs under `test_img/out/`.
- `python -m py_compile results_SIFT_SURF_RIFT_SCB_HOPC.py common/*.py template_based/*.py TAMM_clean/*.py feature_based/*.py`: performs a fast syntax check.
- `python TAMM_clean/main_TAMM.py`: TAMM-specific runner; update the hard-coded input/output paths before use.

Expected Python dependencies include `numpy`, `opencv-python`, `scipy`, `rasterio`, `torch`, `joblib`, `matplotlib`, `kornia`, and `pandas`. Some feature-based methods also require MATLAB Engine for Python.

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Prefer `snake_case` for functions, variables, and module names; use `PascalCase` for classes. Keep algorithm names consistent with existing lowercase method keys such as `tamm`, `cfog`, `sift`, and `rift`. Avoid adding absolute local paths; prefer paths relative to `TWMM/`.

## Testing Guidelines

No automated test suite is currently present. Validate changes by running the demo on `TWMM/test_img/` and inspecting generated `homo/*.csv`, `warp_thermal/`, `mosaic/`, and `matchpoints_img/` outputs. For new tests, create a `tests/` directory at the repository root and name files `test_<module>.py`.

## Commit & Pull Request Guidelines

No established commit history is available in this checkout. Use short, imperative commit messages with an optional scope, for example `Fix TAMM output path handling` or `Add CFOG validation script`. Pull requests should describe the method affected, list commands run, mention required datasets or MATLAB dependencies, and include before/after output images when registration behavior changes.

## Security & Configuration Tips

Do not commit large generated outputs, private datasets, machine-specific absolute paths, or local IDE metadata. Document any new external dependency in the PR until a formal requirements file is added.
