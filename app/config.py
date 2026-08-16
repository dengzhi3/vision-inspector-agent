"""集中管理配置：模型路径、阈值、设备等不写死在业务代码中。

所有参数都有默认值，并支持通过环境变量覆盖，便于部署调整：

    VISION_MODEL_PATH   模型权重路径（默认 models/best.pt）
    VISION_CONFIDENCE   置信度阈值（默认 0.25）
    VISION_IOU          NMS IoU 阈值（默认 0.45）
    VISION_IMAGE_SIZE   推理输入尺寸（默认 640）
    VISION_DEVICE       推理设备：cpu / 0 / cuda:0（默认 None，自动选择）
    VISION_OUTPUT_DIR   批量预测结果输出目录（默认 outputs/）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 项目根目录（app/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        )

