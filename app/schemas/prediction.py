"""统一的数据结构：推理结果全部以这里的 schema 返回。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """GET /health 的响应结构。"""

    status: str = "ok"

    model_config = ConfigDict(extra="forbid")


class Detection(BaseModel):
    """单个检测框。"""
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

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


class ModelFileInfo(BaseModel):
    """模型权重文件信息。"""

    path: str = Field(
        description="Path of the model weights file",
    )
    exists: bool = Field(
        description="Whether the model weights file exists",
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="File size in bytes, or None when the file does not exist",
    )


class InferenceConfig(BaseModel):
    """推理相关配置。"""

    confidence_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence threshold used for inference",
    )
    iou_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="IoU threshold used for NMS",
    )
    image_size: int = Field(
        gt=0,
        description="Inference input image size",
    )
    device: str | None = Field(
        default=None,
        description="Inference device (cpu / cuda:N / None for auto)",
    )


class ModelInfoResponse(BaseModel):
    """GET /model/info 的响应结构。"""
    model_config = ConfigDict(extra="forbid")

    model: ModelFileInfo = Field(
        description="Model weights file information",
    )
    inference: InferenceConfig = Field(
        description="Inference settings",
    )
    output_dir: str = Field(
        description="Directory for batch prediction outputs",
    )


class PredictionResponse(DetectionResult):
    """POST /predictions 的响应结构，字段与 DetectionResult 一致。"""
    model_config = ConfigDict(extra="forbid")


class BatchItem(BaseModel):
    """批量预测中单个文件的预测结果。"""
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(
        description="Original uploaded filename",
    )
    result: PredictionResponse = Field(
        description="Prediction result for this file",
    )


class BatchResponse(BaseModel):
    """POST /predictions/batch 的响应结构。"""
    model_config = ConfigDict(extra="forbid")

    results: list[BatchItem] = Field(
        description="Prediction results in upload order",
    )
    total: int = Field(
        ge=0,
        description="Number of images processed",
    )


class ErrorResponse(BaseModel):
    """统一错误响应结构。"""

    error_code: str = Field(
        description="Machine-readable error code",
    )
    message: str = Field(
        description="Human-readable error message",
    )


class TaskStatus(str, Enum):
    """异步任务的运行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatusResponse(BaseModel):
    """任务状态查询的响应结构。"""
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(
        description="Task identifier",
    )
    status: TaskStatus = Field(
        description="Current task status",
    )
    filename: str | None = Field(
        default=None,
        description="Original uploaded filename",
    )
    result: PredictionResponse | None = Field(
        default=None,
        description="Prediction result when the task is completed",
    )
    error: ErrorResponse | None = Field(
        default=None,
        description="Error info when the task failed",
    )


class PredictionHistory(BaseModel):
    """数据库持久化的一条历史预测记录（GET / 查询历史）。"""
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        ge=1,
        description="prediction_tasks 表主键，可用于查询详情",
    )
    status: TaskStatus = Field(
        description="Prediction task status",
    )
    filename: str | None = Field(
        default=None,
        description="Original uploaded filename, None when the task has no saved image",
    )
    model_name: str | None = Field(
        default=None,
        description="Model name used for the prediction",
    )
    inference_time_ms: float | None = Field(
        default=None,
        ge=0,
        description="Inference time in milliseconds",
    )
    created_at: str = Field(
        min_length=1,
        description="Task creation time (ISO 8601 UTC)",
    )
    completed_at: str | None = Field(
        default=None,
        description="Task completion time (ISO 8601 UTC)",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message when the task failed",
    )
