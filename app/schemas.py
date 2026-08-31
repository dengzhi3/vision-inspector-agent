"""统一的数据结构：推理结果全部以这里的 schema 返回。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str = "ok"

@dataclass
class Detection:
    """单个检测框。"""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2]，整数像素坐标


@dataclass
class DetectionResult:
    """单张图片的推理结果。"""

    image_width: int
    image_height: int
    inference_time_ms: float
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为与约定一致的 JSON 结构。"""
        return asdict(self)

