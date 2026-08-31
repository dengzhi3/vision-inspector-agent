"""统一的数据结构：推理结果全部以这里的 schema 返回。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"


class Detection(BaseModel):
    """单个检测框。"""

    class_id: int = Field(
        ge=0,
        description="Detected class ID",
    )
    class_name: str = Field(
        min_length=1,
        description="Detected class name",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the detection",
    )
    bbox: list[int] = Field(
        description="Bounding box coordinates the in the format of [x1, y1, x2, y2]",
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def validate_bbox(cls, bbox: list[int]) -> list[int]:
        """确保 bbox 是长度为 4 的整数列表，且满足 [x1, y1, x2, y2] 的坐标约束。"""
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox 必须是长度为 4 的列表")
        if not all(isinstance(coord, int) for coord in bbox):
            raise ValueError("bbox 中的坐标必须是整数")
        x1, y1, x2, y2 = bbox

        if x1 < 0 or y1 < 0:
            raise ValueError("bbox coordinates must be non-negative")

        if x2 <= x1:
            raise ValueError("x2 must be greater than x1")

        if y2 <= y1:
            raise ValueError("y2 must be greater than y1")
        return bbox


class DetectionResult(BaseModel):
    """单张图片的推理结果。"""

    image_width: int = Field(
        gt=0,
        description="Width of the input image in pixels",
    )
    image_height: int = Field(
        gt=0,
        description="Height of the input image in pixels",
    )
    inference_time_ms: float = Field(
        ge=0.0,
        description="Inference time of the prediction in milliseconds",
    )
    detections: list[Detection] = Field(
        default_factory=list,
        description="Detected objects in the image",
    )

    def to_dict(self) -> dict:
        """转换为与约定一致的 JSON 结构。"""
        return self.model_dump()
