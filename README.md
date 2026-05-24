# 红外-可见光视频配准流程

本项目用于处理成对的红外视频和可见光视频。流程包括：同步抽帧、基于单目标定文件去畸变、将可见光图像 resize 到红外分辨率 `640x512`、运行 RIFT 多模态配准、输出 warp 后的可见光图像和配准效果预览图。

## 目录结构准备

每一组视频数据建议单独放在一个目录中，目录内至少包含一个红外视频和一个可见光视频。默认情况下，总控脚本会自动识别文件名中包含 `infrared` 和 `visible` 的视频。

```text
I-R-Reg/
├── data/
│   └── video_data_20260521/
│       ├── infrared_20260521_204552.mp4
│       ├── visible_20260521_204552.mp4
│       ├── frames/
│       │   ├── infrared/
│       │   └── visible/
│       ├── frames_undistorted/
│       │   ├── infrared/
│       │   └── visible/
│       └── rift_registration/
│           ├── warped_rgb/
│           ├── preview/
│           └── homographies.csv
├── RIFT-Multimodal-matching-python/
├── RIFT2-multimodal-matching-rotation-python/
├── scripts/
│   ├── extract_video_frames.py
│   ├── undistort_extracted_frames.py
│   ├── register_frames_rift.py
│   ├── register_frames_rift2.py
│   └── run_video_rift_pipeline.py
├── config/
│   ├── ir_mono.yaml
│   └── rgb_mono.yaml
└── TWMM/
```

其中 `frames/`、`frames_undistorted/` 和 `rift_registration/` 是运行脚本后生成的目录，不需要手动创建。

## 使用 uv 管理环境

创建并激活虚拟环境：

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
```

安装当前流程需要的 Python 包：

```bash
uv pip install --python .venv/bin/python \
  opencv-python numpy scipy joblib PyYAML tqdm
```

如果 uv 默认缓存目录没有写权限，可以使用项目内缓存：

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python \
  opencv-python numpy scipy joblib PyYAML tqdm
```

当前已使用的主要依赖版本：

```text
opencv-python==4.13.0
numpy==2.2.6
scipy==1.15.3
joblib==1.5.3
PyYAML==6.0.3
tqdm==4.67.3
```

导出当前环境依赖：

```bash
UV_CACHE_DIR=.uv-cache uv pip freeze --python .venv/bin/python > requirements.txt
```

## 运行总控脚本

推荐使用总控脚本完成“视频抽帧 + 去畸变 + RIFT 配准”的完整流程。

先处理 1 对图片做快速验证：

```bash
.venv/bin/python scripts/run_video_rift_pipeline.py data/video_data_20260521 --max-pairs 1
```

确认效果后，处理全部帧对：

```bash
.venv/bin/python scripts/run_video_rift_pipeline.py data/video_data_20260521 --max-pairs 0
```

使用新 clone 的 RIFT2 后端处理全部帧对，结果默认写入 `rift2_registration/`：

```bash
.venv/bin/python scripts/run_video_rift_pipeline.py data/video_data_20260521 \
  --register-backend rift2 \
  --max-pairs 0 \
  --overwrite
```

常用参数：

```bash
--step 10                  # 每 10 帧抽取 1 张图片
--overwrite                # 覆盖已有抽帧和配准结果，强制重新计算
--skip-extract             # 跳过抽帧，继续去畸变和配准
--skip-register            # 只抽帧和去畸变，不运行配准
--register-backend rift2   # 使用新 RIFT2 后端；默认是 rift
--no-fallback-h            # 关闭失败帧复用最近成功 H 的默认策略
--frames-dir PATH          # 指定抽帧输出目录
--undistorted-frames-dir PATH # 指定去畸变输出目录
--registration-dir PATH    # 指定配准结果输出目录
--ir-config PATH           # 指定红外单目标定文件
--rgb-config PATH          # 指定可见光单目标定文件
--infrared-name FILE       # 手动指定红外视频文件名
--visible-name FILE        # 手动指定可见光视频文件名
```

如果视频文件名不能被自动识别，可以显式指定：

```bash
.venv/bin/python scripts/run_video_rift_pipeline.py /path/to/video_batch \
  --infrared-name infrared.mp4 \
  --visible-name visible.mp4 \
  --frames-dir /path/to/frames \
  --undistorted-frames-dir /path/to/frames_undistorted \
  --registration-dir /path/to/rift_output \
  --max-pairs 0
```

生成未配准的直接 overlay 对比基线：

```bash
.venv/bin/python scripts/make_raw_overlay_baseline.py \
  --input-root data/video_data_20260521/frames_undistorted \
  --output-dir data/video_data_20260521/rift_registration/raw_overlay \
  --max-pairs 0 \
  --overwrite
```

## 输出结果

抽帧结果：

```text
frames/infrared/infrared_000000.jpg
frames/visible/visible_000000.jpg
```

去畸变结果：

```text
frames_undistorted/infrared/infrared_000000.jpg
frames_undistorted/visible/visible_000000.jpg
```

RIFT 配准结果：

```text
rift_registration/warped_rgb/visible_warped_000000.jpg
rift_registration/preview/preview_000000.jpg
rift_registration/inlier_matches/inliers_000000.jpg
rift_registration/raw_overlay/raw_overlay_000000.jpg
rift_registration/homographies.csv
```

RIFT2 后端输出结构相同，但默认目录是 `rift2_registration/`，便于和原 RIFT 结果对比。

总控脚本默认使用 `frames_undistorted/` 作为配准输入。`warped_rgb/` 中保存的是 warp 到红外坐标系下的可见光 RGB 图像。`preview/` 中保存的是 2x2 配准效果图，包含 overlay、棋盘格、warp 后可见光图和红外原图。`inlier_matches/` 中保存红外与可见光的 RIFT/RIFT2+RANSAC 内点连线图，默认最多显示 100 条内点连线。`raw_overlay/` 中保存未配准前的红外与可见光直接叠加图，可作为配准效果对比基线。`homographies.csv` 记录每帧的状态、匹配点数、内点数和实际用于输出的可见光到红外单应性矩阵。

配准阶段默认启用质量门控回退策略：如果当前帧匹配不足、单应性估计失败或 H 未通过质量门控，会复用当前帧之前最近一次 `ok` 或 `fallback` 的 H 生成输出，并将该帧记录为 `status=fallback`。如果之前没有可用 H，则记录为 `status=failed`，并输出 resize 后的原始可见光图。`fallback` 帧的内点连线图展示的是当前帧本次 RANSAC 结果，用于诊断当前匹配质量，不代表最终用于 warp 的 fallback H。需要严格复现旧逻辑时，可加 `--no-fallback-h`。
