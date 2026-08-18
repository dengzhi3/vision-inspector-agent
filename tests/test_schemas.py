"""app.schemas 的单元测试。"""

from __future__ import annotations

from dataclasses import asdict

from app.schemas import Detection, DetectionResult


def test_detection_creation():
    """Detection 应保存单个检测框的全部字段。"""
    detection = Detection(
        class_id=0,
        class_name="crack",
        confidence=0.91,
        bbox=[120, 80, 600, 420],
    )

    assert detection.class_id == 0
    assert detection.class_name == "crack"
    assert detection.confidence == 0.91
    assert detection.bbox == [120, 80, 600, 420]

    # dataclass 语义：相同字段构造的实例应相等，asdict 可转回普通字典
    same = Detection(class_id=0, class_name="crack", confidence=0.91, bbox=[120, 80, 600, 420])
    assert detection == same
    assert asdict(detection) == {
        "class_id": 0,
        "class_name": "crack",
        "confidence": 0.91,
        "bbox": [120, 80, 600, 420],
    }


def test_detection_result_creation():
    """DetectionResult 应保存图片信息与检测列表，detections 缺省为空列表。"""
    detection = Detection(
        class_id=1,
        class_name="leak",
        confidence=0.8765,
        bbox=[50, 60, 200, 300],
    )
    result = DetectionResult(
        image_width=640,
        image_height=480,
        inference_time_ms=45.0,
        detections=[detection],
    )

    assert result.image_width == 640
    assert result.image_height == 480
    assert result.inference_time_ms == 45.0
    assert result.detections == [detection]

    # 未提供 detections 时应默认空列表，而不是共享可变默认值
    empty = DetectionResult(image_width=320, image_height=240, inference_time_ms=10.0)
    assert empty.detections == []

    assert result.to_dict() == {
        "image_width": 640,
        "image_height": 480,
        "inference_time_ms": 45.0,
        "detections": [
            {
                "class_id": 1,
                "class_name": "leak",
                "confidence": 0.8765,
                "bbox": [50, 60, 200, 300],
            }
        ],
    }


def test_bbox_structure():
    """bbox 应为 [x1, y1, x2, y2] 的整数像素坐标，且 x1<=x2、y1<=y2。"""
    detection = Detection(
        class_id=0,
        class_name="crack",
        confidence=0.91,
        bbox=[120, 80, 600, 420],
    )

    x1, y1, x2, y2 = detection.bbox

    assert len(detection.bbox) == 4
    assert all(isinstance(coord, int) for coord in detection.bbox)
    assert x1 <= x2
    assert y1 <= y2
    assert detection.bbox == [120, 80, 600, 420]
