"""Pydantic 模型包：统一从这里导入各 schema。"""

from app.schemas.prediction import (
    BatchItem,
    BatchResponse,
    Detection,
    DetectionResult,
    ErrorResponse,
    HealthResponse,
    InferenceConfig,
    ModelFileInfo,
    ModelInfoResponse,
    PredictionResponse,
    PredictionHistory,
    TaskStatus,
    TaskStatusResponse,
)

__all__ = [
    "BatchItem",
    "BatchResponse",
    "Detection",
    "DetectionResult",
    "ErrorResponse",
    "HealthResponse",
    "InferenceConfig",
    "ModelFileInfo",
    "ModelInfoResponse",
    "PredictionResponse",
    "PredictionHistory",
    "TaskStatus",
    "TaskStatusResponse",
]
