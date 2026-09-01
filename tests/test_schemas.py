"""app.schemas 的单元测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    Detection,
    DetectionResult,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)


@pytest.fixture
def detection_kwargs() -> dict[str, object]:
    """构造合法 Detection 的默认字段值。"""
    return {
        "class_id": 0,
        "class_name": "crack",
        "confidence": 0.91,
        "bbox": [120, 80, 600, 420],
    }


@pytest.fixture
def detection(detection_kwargs) -> Detection:
    """一个合法的 Detection 实例。"""
    return Detection(**detection_kwargs)


@pytest.fixture
def detection_result_kwargs() -> dict[str, object]:
    """构造合法 DetectionResult 的默认字段值。"""
    return {
        "image_width": 640,
        "image_height": 480,
        "inference_time_ms": 45.0,
    }


@pytest.fixture
def detection_result(detection_result_kwargs, detection) -> DetectionResult:
    """包含一个检测框的 DetectionResult 实例。"""
    return DetectionResult(**detection_result_kwargs, detections=[detection])


@pytest.fixture
def empty_result(detection_result_kwargs) -> DetectionResult:
    """未显式提供 detections 的 DetectionResult 实例。"""
    return DetectionResult(**detection_result_kwargs)


def test_detection_creation(detection, detection_kwargs):
    """Detection 应保存单个检测框的全部字段。"""
    assert detection.class_id == 0
    assert detection.class_name == "crack"
    assert detection.confidence == 0.91
    assert detection.bbox == [120, 80, 600, 420]

    # Pydantic 语义：相同字段构造的实例应相等，model_dump() 可转为普通字典
    same = Detection(**detection_kwargs)
    assert detection == same
    assert detection.model_dump() == detection_kwargs


def test_detection_result_creation(detection_result, detection, empty_result):
    """DetectionResult 应保存图片信息与检测列表，detections 缺省为空列表。"""
    assert detection_result.image_width == 640
    assert detection_result.image_height == 480
    assert detection_result.inference_time_ms == 45.0
    assert detection_result.detections == [detection]

    # 未提供 detections 时默认为空列表，而不是共享的可变默认值
    assert empty_result.detections == []

    assert detection_result.to_dict() == {
        "image_width": 640,
        "image_height": 480,
        "inference_time_ms": 45.0,
        "detections": [
            {
                "class_id": 0,
                "class_name": "crack",
                "confidence": 0.91,
                "bbox": [120, 80, 600, 420],
            }
        ],
    }


def test_bbox_structure(detection):
    """bbox 应为 [x1, y1, x2, y2] 的整数像素坐标，且 x1<x2、y1<y2。"""
    x1, y1, x2, y2 = detection.bbox

    assert len(detection.bbox) == 4
    assert all(isinstance(coord, int) for coord in detection.bbox)
    assert x1 < x2
    assert y1 < y2
    assert detection.bbox == [120, 80, 600, 420]


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 10, 10],
        [120, 80, 600, 420],
        [0, 0, 1, 1],
    ],
)
def test_bbox_validation_valid(detection_kwargs, bbox):
    """合法的 bbox 应通过 validate_bbox 并保持原值。"""
    detection = Detection(**{**detection_kwargs, "bbox": bbox})

    assert detection.bbox == bbox


@pytest.mark.parametrize(
    "bbox",
    [
        [],                          # 长度不足
        [1, 2, 3],                   # 长度不足
        [0, 0, 10, 10, 20],          # 长度超限
        "abcd",                      # 不是列表
        [0, 0.5, 10, 10],            # 坐标不是整数
        [-1, 0, 10, 10],             # x1 为负
        [0, -1, 10, 10],             # y1 为负
        [10, 0, 10, 10],             # x2 不大于 x1
        [10, 0, 5, 10],              # x2 小于 x1
        [0, 10, 10, 10],             # y2 不大于 y1
        [0, 10, 10, 5],              # y2 小于 y1
    ],
)
def test_bbox_validation_invalid(detection_kwargs, bbox):
    """非法的 bbox 应由 validate_bbox 拒绝并触发 ValidationError。"""
    with pytest.raises(ValidationError):
        Detection(**{**detection_kwargs, "bbox": bbox})


@pytest.mark.parametrize(
    "overrides",
    [
        {"class_id": -1},          # class_id 必须 >= 0
        {"class_name": ""},        # class_name 至少 1 个字符
        {"confidence": -0.01},     # confidence 必须 >= 0
        {"confidence": 1.01},      # confidence 必须 <= 1
    ],
)
def test_detection_validation(detection_kwargs, overrides):
    """Detection 的 Field 约束应拒绝非法字段值。"""
    with pytest.raises(ValidationError):
        Detection(**{**detection_kwargs, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_width": 0},            # image_width 必须 > 0
        {"image_width": -1},
        {"image_height": 0},           # image_height 必须 > 0
        {"image_height": -1},
        {"inference_time_ms": -0.1},   # inference_time_ms 必须 >= 0
    ],
)
def test_detection_result_validation(detection_result_kwargs, overrides):
    """DetectionResult 的 Field 约束应拒绝非法字段值。"""
    with pytest.raises(ValidationError):
        DetectionResult(**{**detection_result_kwargs, **overrides})


@pytest.mark.parametrize(
    ("model", "valid_kwargs"),
    [
        (HealthResponse, {}),
        (
            Detection,
            {
                "class_id": 0,
                "class_name": "crack",
                "confidence": 0.91,
                "bbox": [120, 80, 600, 420],
            },
        ),
        (
            DetectionResult,
            {
                "image_width": 640,
                "image_height": 480,
                "inference_time_ms": 45.0,
            },
        ),
        (
            PredictionResponse,
            {
                "image_width": 640,
                "image_height": 480,
                "inference_time_ms": 45.0,
            },
        ),
        (
            ModelInfoResponse,
            {
                "model": {"path": "models/yolo.pt", "exists": True},
                "inference": {
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45,
                    "image_size": 640,
                },
                "output_dir": "outputs",
            },
        ),
    ],
    ids=[
        "HealthResponse",
        "Detection",
        "DetectionResult",
        "PredictionResponse",
        "ModelInfoResponse",
    ],
)
def test_extra_fields_forbidden(model, valid_kwargs):
    """配置了 model_config = ConfigDict(extra="forbid") 的模型应拒绝未知字段。"""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model(**valid_kwargs, unexpected_field="boom")


def test_detection_result_rejects_unknown_nested_fields(detection_result_kwargs):
    """extra="forbid" 同样作用于嵌套模型，detections 内的未知字段也应被拒绝。"""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DetectionResult(
            **detection_result_kwargs,
            detections=[
                {
                    "class_id": 0,
                    "class_name": "crack",
                    "confidence": 0.91,
                    "bbox": [120, 80, 600, 420],
                    "unexpected_field": "boom",
                }
            ],
        )
