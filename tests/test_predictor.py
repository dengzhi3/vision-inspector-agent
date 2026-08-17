"""app.predictor 的单元测试。

使用 mock 模型代替真实推理，保证测试快速、稳定且不依赖模型文件。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

from app.config import Settings
from app.predictor import InvalidImageError, predict_image
from app.schemas import Detection, DetectionResult


class _TolistList(list):
    """模拟 torch/numpy 张量的 tolist() 接口。"""

    def tolist(self) -> list:
        return list(self)


class _FakeBoxes:
    """模拟检测框容器，支持 len() 与 xyxy/conf/cls 三个属性。"""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = _TolistList(xyxy)
        self.conf = _TolistList(conf)
        self.cls = _TolistList(cls)

    def __len__(self) -> int:
        return len(self.xyxy)


def _save_image(path: Path, width: int = 320, height: int = 240) -> Path:
    Image.new("RGB", (width, height), "white").save(path)
    return path


def _make_result(orig_shape=(240, 320), boxes=None):
    """构造模型返回结果的最小替身。"""
    return SimpleNamespace(orig_shape=orig_shape, boxes=boxes)


@pytest.fixture
def valid_image(tmp_path):
    """一张有效的临时 PNG 图片路径（320x240）。"""
    return _save_image(tmp_path / "valid.png", width=320, height=240)


@pytest.fixture
def large_image(tmp_path):
    """一张较大的有效临时 PNG 图片路径（640x480）。"""
    return _save_image(tmp_path / "large.png", width=640, height=480)


@pytest.fixture
def model(monkeypatch):
    """打桩 load_model 并返回可编程的 Mock 模型。"""
    fake = mock.Mock()
    monkeypatch.setattr("app.predictor.load_model", lambda *args, **kwargs: fake)
    return fake


@pytest.fixture
def empty_result():
    """无检测目标的推理结果。"""
    return _make_result(orig_shape=(240, 320), boxes=None)


@pytest.fixture
def result_with_boxes():
    """包含两个检测框的推理结果。"""
    boxes = _FakeBoxes(
        xyxy=[[10.4, 20.6, 100.2, 200.8], [50.4, 60.6, 80.4, 90.6]],
        conf=[0.87654, 0.123456],
        cls=[0, 1],
    )
    return _make_result(orig_shape=(240, 320), boxes=boxes)


def test_predict_valid_image(valid_image, model, empty_result):
    """有效图片应返回结构完整的 DetectionResult，无检测目标时 detections 为空。"""
    model.predict.return_value = [empty_result]

    prediction = predict_image(valid_image)

    assert isinstance(prediction, DetectionResult)
    assert prediction.image_width == 320
    assert prediction.image_height == 240
    assert isinstance(prediction.inference_time_ms, float)
    assert prediction.inference_time_ms >= 0
    assert prediction.detections == []


def test_predict_missing_image(tmp_path, monkeypatch):
    """不存在的图片路径应抛出 InvalidImageError，且不会尝试加载模型。"""
    missing = tmp_path / "missing.jpg"
    assert not missing.exists()

    loader = mock.Mock()
    monkeypatch.setattr("app.predictor.load_model", loader)

    with pytest.raises(InvalidImageError, match="不存在"):
        predict_image(missing)

    loader.assert_not_called()


def test_predict_corrupted_image(tmp_path, model):
    """无法解码的图片应抛出 InvalidImageError。"""
    image = tmp_path / "corrupted.jpg"
    image.write_bytes(b"this is not a real jpeg")
    model.predict.side_effect = RuntimeError("decode failed")

    with pytest.raises(InvalidImageError, match="无法解码"):
        predict_image(image)


def test_prediction_image_size(large_image, model):
    """配置的 image_size 应作为 imgsz 传给模型，结果应反映图片原始尺寸。"""
    model.predict.return_value = [_make_result(orig_shape=(480, 640), boxes=None)]

    settings = Settings(image_size=320, device="cpu")
    prediction = predict_image(large_image, settings=settings)

    assert model.predict.call_args.kwargs["imgsz"] == 320
    assert model.predict.call_args.kwargs["device"] == "cpu"
    assert prediction.image_width == 640
    assert prediction.image_height == 480


@pytest.mark.parametrize("threshold", [0.05, 0.25, 0.5, 0.9])
def test_prediction_confidence_threshold(valid_image, model, empty_result, threshold):
    """置信度阈值应作为 conf 传给模型。"""
    model.predict.return_value = [empty_result]
    settings = Settings(confidence_threshold=threshold)

    predict_image(valid_image, settings=settings)

    assert model.predict.call_args.kwargs["conf"] == threshold


def test_prediction_detections(valid_image, model, result_with_boxes):
    """多个检测框应解析为 Detection，坐标取整、置信度保留 4 位小数。"""
    model.names = {0: "crack", 1: "leak"}
    model.predict.return_value = [result_with_boxes]

    prediction = predict_image(valid_image)

    assert len(prediction.detections) == 2
    first, second = prediction.detections

    assert isinstance(first, Detection)
    assert first.class_id == 0
    assert first.class_name == "crack"
    assert first.confidence == 0.8765
    assert first.bbox == [10, 21, 100, 201]

    assert second.class_id == 1
    assert second.class_name == "leak"
    assert second.confidence == 0.1235
    assert second.bbox == [50, 61, 80, 91]
