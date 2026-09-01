"""app.api 的接口测试。

使用 FastAPI TestClient 直接调用路由，并通过 monkeypatch 替换
app.api.predict_image，避免加载真实 YOLO 模型，保证测试快速、稳定。
预测器内部的异常行为（图片解码失败等）已由 tests/test_predictor.py 覆盖。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import app
from app.config import Settings
from app.model_loader import ModelNotFoundError
from app.predictor import InvalidImageError
from app.schemas import DetectionResult, ErrorResponse


def _png_bytes(width: int = 320, height: int = 240) -> bytes:
    """生成一张有效 PNG 图片的字节内容。"""
    buf = BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _assert_error_response(
    resp, status_code: int, error_code: str, message: str
) -> None:
    """断言响应符合 ErrorResponse schema，且字段内容与预期一致。"""
    body = resp.json()

    assert resp.status_code == status_code
    # ErrorResponse 只应有 error_code 与 message 两个字段，不允许多余字段
    assert set(body) == {"error_code", "message"}
    parsed = ErrorResponse.model_validate(body)
    assert parsed.error_code == error_code
    assert parsed.message == message


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_png() -> bytes:
    return _png_bytes()


def test_health(client):
    """GET /health 应返回 200 与 status=ok。"""
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info(client, monkeypatch, tmp_path):
    """GET /model/info 应返回由环境变量构造的模型与推理配置。"""
    model_file = tmp_path / "best.pt"
    model_file.write_bytes(b"fake weights")

    monkeypatch.setenv("VISION_MODEL_PATH", str(model_file))
    monkeypatch.setenv("VISION_CONFIDENCE", "0.42")
    monkeypatch.setenv("VISION_IOU", "0.6")
    monkeypatch.setenv("VISION_IMAGE_SIZE", "320")
    monkeypatch.setenv("VISION_DEVICE", "cpu")
    output_dir = tmp_path / "outputs"
    monkeypatch.setenv("VISION_OUTPUT_DIR", str(output_dir))

    resp = client.get("/model/info")

    assert resp.status_code == 200
    assert resp.json() == {
        "model": {
            "path": str(model_file),
            "exists": True,
            "size_bytes": len(b"fake weights"),
        },
        "inference": {
            "confidence_threshold": 0.42,
            "iou_threshold": 0.6,
            "image_size": 320,
            "device": "cpu",
        },
        "output_dir": str(output_dir),
    }


def test_prediction_valid_image(client, valid_png, monkeypatch):
    """上传有效图片应返回 DetectionResult 对应的 JSON，并清理临时文件。"""
    fake = mock.Mock(
        return_value=DetectionResult(
            image_width=320,
            image_height=240,
            inference_time_ms=12.3,
        )
    )
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions",
        files={"file": ("valid.png", valid_png, "image/png")},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "image_width": 320,
        "image_height": 240,
        "inference_time_ms": 12.3,
        "detections": [],
    }

    fake.assert_called_once()
    assert isinstance(fake.call_args.args[0], Path)
    assert fake.call_args.args[0].suffix == ".png"
    assert isinstance(fake.call_args.kwargs["settings"], Settings)
    # 推理返回后临时文件应已被清理
    assert not fake.call_args.args[0].exists()


def test_prediction_invalid_file(client, monkeypatch):
    """上传无法解码的图片时，API 应返回 400，响应体符合 ErrorResponse schema。"""

    def _raise_invalid(*args, **kwargs):
        raise InvalidImageError("图片无法解码或推理失败")

    monkeypatch.setattr("app.api.predict_image", _raise_invalid)

    resp = client.post(
        "/predictions",
        files={"file": ("broken.png", b"this is not a png", "image/png")},
    )

    _assert_error_response(resp, 400, "INVALID_IMAGE", "图片无法解码或推理失败")


def test_prediction_model_not_found(client, monkeypatch):
    """模型文件缺失时，API 应返回 500，响应体符合 ErrorResponse schema。"""

    def _raise_model(*args, **kwargs):
        raise ModelNotFoundError("模型文件不存在")

    monkeypatch.setattr("app.api.predict_image", _raise_model)

    resp = client.post(
        "/predictions",
        files={"file": ("valid.png", _png_bytes(), "image/png")},
    )

    _assert_error_response(resp, 500, "MODEL_NOT_FOUND", "模型文件不存在")


def test_openapi_declares_error_response():
    """OpenAPI 中 /predictions 的 400/500 响应应引用 ErrorResponse schema。"""
    schema = app.openapi()
    responses = schema["paths"]["/predictions"]["post"]["responses"]

    assert "400" in responses
    assert "500" in responses
    assert responses["400"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    assert responses["500"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    assert "ErrorResponse" in schema["components"]["schemas"]
