"""预测逻辑：单张图片与目录批量预测，返回统一结构，不直接 print。"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.config import Settings
from app.schemas.prediction import Detection, DetectionResult
from app.vision.model_loader import load_model

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class InvalidImageError(ValueError):
    """图片路径无效或图片无法解码时抛出。"""


def _validate_image_path(image_path: str | Path) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise InvalidImageError(f"图片不存在: {path}")
    if not path.is_file():
        raise InvalidImageError(f"路径不是文件: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        raise InvalidImageError(
            f"不支持的图片格式: {path.suffix!r}（支持: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}）"
        )
    return path


def predict_image(image_path: str | Path, settings: Settings | None = None) -> DetectionResult:
    """对单张图片执行推理，返回统一结构 DetectionResult。"""
    settings = settings or Settings.from_env()
    path = _validate_image_path(image_path)

    model = load_model(settings.model_path)

    start = time.perf_counter()
    try:
        result = model.predict(
            source=str(path),
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            imgsz=settings.image_size,
            device=settings.device,
            verbose=False,
        )[0]
    except Exception as exc:
        raise InvalidImageError(f"图片无法解码或推理失败: {path}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    height, width = result.orig_shape
    detections: list[Detection] = []
    if result.boxes is not None and len(result.boxes) > 0:
        names = model.names
        for box, conf, cls_id in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=str(names[int(cls_id)]),
                    confidence=round(float(conf), 4),
                    bbox=[round(coord) for coord in box],
                )
            )

    return DetectionResult(
        image_width=int(width),
        image_height=int(height),
        inference_time_ms=round(elapsed_ms, 1),
        detections=detections,
    )


def predict_directory(
    image_dir: str | Path, settings: Settings | None = None
) -> dict[str, DetectionResult]:
    """批量预测目录下所有支持的图片，返回 {文件名: DetectionResult}。"""
    directory = Path(image_dir)
    if not directory.is_dir():
        raise InvalidImageError(f"目录不存在: {directory}")

    images = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )
    if not images:
        raise InvalidImageError(f"目录中没有支持的图片: {directory}")

    results: dict[str, DetectionResult] = {}
    for image in images:
        results[image.name] = predict_image(image, settings=settings)
    return results
