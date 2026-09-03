"""集中管理配置：模型路径、阈值、设备、上传限制等不写死在业务代码中。

所有参数都有默认值，并支持通过环境变量覆盖，便于部署调整：

    VISION_MODEL_PATH           模型权重路径（默认 models/best.pt）
    VISION_CONFIDENCE           置信度阈值（默认 0.25）
    VISION_IOU                  NMS IoU 阈值（默认 0.45）
    VISION_IMAGE_SIZE           推理输入尺寸（默认 640）
    VISION_DEVICE               推理设备：cpu / 0 / cuda:0（默认 None，自动选择）
    VISION_OUTPUT_DIR           批量预测结果输出目录（默认 outputs/）
    VISION_MAX_FILE_SIZE_MB     单张上传图片大小上限（默认 10 MB）
    VISION_PREDICTION_TIMEOUT   单张图片预测超时（秒，默认 30）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 项目根目录（app/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 单张上传图片的大小上限（MB），可通过 VISION_MAX_FILE_SIZE_MB 覆盖
DEFAULT_MAX_FILE_SIZE_MB = 10

# API 允许上传的图片 MIME 类型（与 vision.predictor.SUPPORTED_IMAGE_EXTS 对应）
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}

# 批量接口单次上传的图片数量上限
MAX_BATCH_FILES = 10


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass
class Settings:
    """推理相关配置。"""

    model_path: Path = PROJECT_ROOT / "models" / "best.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str | None = None  # None 表示让 ultralytics 自动选择设备
    output_dir: Path = PROJECT_ROOT / "outputs"
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_MB * 1024 * 1024  # 单张上传图片大小上限（字节）
    prediction_timeout_seconds: float = 30.0  # 单张图片预测超时（秒）

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量构造配置，未设置的环境变量使用默认值。"""
        return cls(
            model_path=Path(os.getenv("VISION_MODEL_PATH", str(cls.model_path))),
            confidence_threshold=_env_float("VISION_CONFIDENCE", cls.confidence_threshold),
            iou_threshold=_env_float("VISION_IOU", cls.iou_threshold),
            image_size=_env_int("VISION_IMAGE_SIZE", cls.image_size),
            device=os.getenv("VISION_DEVICE") or None,
            output_dir=Path(os.getenv("VISION_OUTPUT_DIR", str(cls.output_dir))),
            max_file_size_bytes=(
                _env_int("VISION_MAX_FILE_SIZE_MB", DEFAULT_MAX_FILE_SIZE_MB) * 1024 * 1024
            ),
            prediction_timeout_seconds=_env_float(
                "VISION_PREDICTION_TIMEOUT", cls.prediction_timeout_seconds
            ),
        )
