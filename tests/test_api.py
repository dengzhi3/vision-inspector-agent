"""app.api 的接口测试。

使用 FastAPI TestClient 直接调用路由，并通过 monkeypatch 替换
app.api.predict_image，避免加载真实 YOLO 模型，保证测试快速、稳定。
预测器内部的异常行为（图片解码失败等）已由 tests/test_predictor.py 覆盖。
"""

from __future__ import annotations

import time
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
from app.schemas import Detection, DetectionResult, ErrorResponse


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
    """OpenAPI 中 /predictions 的 400/500/504 响应应引用 ErrorResponse schema。"""
    schema = app.openapi()
    responses = schema["paths"]["/predictions"]["post"]["responses"]

    assert "400" in responses
    assert "500" in responses
    assert "504" in responses
    assert responses["400"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    assert responses["500"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    assert responses["504"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )
    assert "ErrorResponse" in schema["components"]["schemas"]


def test_prediction_txt_rejected(client):
    """上传 .txt 文件（即使 MIME 声明为图片）应被拒绝并返回 400。"""

    resp = client.post(
        "/predictions",
        files={"file": ("notes.txt", b"hello", "image/png")},
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body["error_code"] == "INVALID_IMAGE"
    assert "不支持的图片格式" in body["message"]
    assert "'.txt'" in body["message"]


def test_prediction_empty_file(client, monkeypatch):
    """上传空文件时推理应失败，API 返回 400 且响应体符合 ErrorResponse schema。"""
    model = mock.Mock()
    model.predict.side_effect = RuntimeError("decode failed")
    monkeypatch.setattr("app.predictor.load_model", lambda *args, **kwargs: model)

    resp = client.post(
        "/predictions",
        files={"file": ("empty.png", b"", "image/png")},
    )

    assert resp.status_code == 400
    body = resp.json()
    assert set(body) == {"error_code", "message"}
    assert body["error_code"] == "INVALID_IMAGE"
    assert body["message"].startswith("图片无法解码或推理失败")


@pytest.mark.parametrize(
    ("filename", "expected_status"),
    [
        # 文件名为空时，multipart 部分被当作普通表单字段，
        # FastAPI 在参数校验阶段直接拒绝（422），不会进入端点逻辑
        ("", 422),
        # 没有图片扩展名时，predictor 的扩展名校验拒绝（400）
        ("no_extension", 400),
    ],
)
def test_prediction_invalid_filename(client, valid_png, filename, expected_status):
    """文件名为空或没有图片扩展名时，请求都应被拒绝。"""

    resp = client.post(
        "/predictions",
        files={"file": (filename, valid_png, "image/png")},
    )

    assert resp.status_code == expected_status
    if expected_status == 400:
        body = resp.json()
        assert body["error_code"] == "INVALID_IMAGE"
        assert "不支持的图片格式" in body["message"]


def test_prediction_wrong_mime_type(client, monkeypatch, valid_png):
    """声明了非图片 MIME 类型（text/plain）时，即使扩展名合法也应返回 400，
    且不会进入推理阶段。"""
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions",
        files={"file": ("valid.png", valid_png, "text/plain")},
    )

    _assert_error_response(resp, 400, "INVALID_IMAGE", "不支持的图片 MIME 类型: 'text/plain'")
    fake.assert_not_called()


def test_prediction_response_structure_with_detections(client, monkeypatch):
    """有检测目标时，响应应包含完整的嵌套结构（class_id/class_name/confidence/bbox）。"""
    fake = mock.Mock(
        return_value=DetectionResult(
            image_width=320,
            image_height=240,
            inference_time_ms=45.6,
            detections=[
                Detection(
                    class_id=0,
                    class_name="crack",
                    confidence=0.8765,
                    bbox=[10, 20, 100, 200],
                )
            ],
        )
    )
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions",
        files={"file": ("valid.png", _png_bytes(), "image/png")},
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "image_width": 320,
        "image_height": 240,
        "inference_time_ms": 45.6,
        "detections": [
            {
                "class_id": 0,
                "class_name": "crack",
                "confidence": 0.8765,
                "bbox": [10, 20, 100, 200],
            }
        ],
    }


def test_batch_prediction_multiple_images(client, monkeypatch):
    """批量上传多张图片应按上传顺序逐张预测，返回每个文件对应的预测结果。"""
    fake = mock.Mock(
        side_effect=[
            DetectionResult(image_width=320, image_height=240, inference_time_ms=12.3),
            DetectionResult(
                image_width=640,
                image_height=480,
                inference_time_ms=25.7,
                detections=[
                    Detection(
                        class_id=0,
                        class_name="crack",
                        confidence=0.9,
                        bbox=[1, 2, 30, 40],
                    )
                ],
            ),
        ]
    )
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", ("a.png", _png_bytes(320, 240), "image/png")),
            ("files", ("b.png", _png_bytes(640, 480), "image/png")),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert [item["filename"] for item in body["results"]] == ["a.png", "b.png"]
    assert body["results"][0]["result"] == {
        "image_width": 320,
        "image_height": 240,
        "inference_time_ms": 12.3,
        "detections": [],
    }
    assert body["results"][1]["result"] == {
        "image_width": 640,
        "image_height": 480,
        "inference_time_ms": 25.7,
        "detections": [
            {
                "class_id": 0,
                "class_name": "crack",
                "confidence": 0.9,
                "bbox": [1, 2, 30, 40],
            }
        ],
    }
    # 逐张顺序预测：每次调用后对应的临时文件都应已清理
    assert fake.call_count == 2
    assert all(not call.args[0].exists() for call in fake.call_args_list)


def test_batch_prediction_rejects_wrong_mime_type(client, monkeypatch):
    """批量上传中任一文件 MIME 类型非法时返回 400，且不会开始推理。"""
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", ("notes.txt", b"hello", "text/plain")),
            ("files", ("a.png", _png_bytes(), "image/png")),
        ],
    )

    _assert_error_response(resp, 400, "INVALID_IMAGE", "不支持的图片 MIME 类型: 'text/plain'")
    fake.assert_not_called()


def test_batch_prediction_propagates_prediction_error(client, monkeypatch):
    """批量预测中某张图片推理失败时，整个请求按单图错误规则返回 400。"""
    fake = mock.Mock(
        side_effect=[
            DetectionResult(image_width=320, image_height=240, inference_time_ms=1.0),
            InvalidImageError("图片无法解码"),
        ]
    )
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", ("a.png", _png_bytes(), "image/png")),
            ("files", ("b.png", _png_bytes(), "image/png")),
        ],
    )

    _assert_error_response(resp, 400, "INVALID_IMAGE", "图片无法解码")


def test_batch_prediction_max_ten_files(client, monkeypatch):
    """恰好 10 张图片应被接受并逐张预测。"""
    fake = mock.Mock(
        return_value=DetectionResult(
            image_width=320,
            image_height=240,
            inference_time_ms=1.0,
        )
    )
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", (f"img{i}.png", _png_bytes(), "image/png"))
            for i in range(10)
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 10
    assert fake.call_count == 10


def test_batch_prediction_rejects_more_than_ten_files(client, monkeypatch):
    """超过 10 张图片应返回 400 TOO_MANY_FILES，且不进入推理。"""
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", (f"img{i}.png", _png_bytes(), "image/png"))
            for i in range(11)
        ],
    )

    _assert_error_response(resp, 400, "TOO_MANY_FILES", "一次最多上传 10 张图片")
    fake.assert_not_called()


@pytest.mark.parametrize("count", [1, 3])
def test_batch_prediction_success_counts(client, monkeypatch, count):
    """批量上传 1 张或 3 张图片都应成功，返回按上传顺序排列的结果。"""
    fake = mock.Mock(
        return_value=DetectionResult(
            image_width=320,
            image_height=240,
            inference_time_ms=1.0,
        )
    )
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", (f"img{i}.png", _png_bytes(), "image/png"))
            for i in range(count)
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == count
    assert [item["filename"] for item in body["results"]] == [
        f"img{i}.png" for i in range(count)
    ]
    assert fake.call_count == count
    # 每张预测完成后临时文件都应已清理
    assert all(not call.args[0].exists() for call in fake.call_args_list)


def test_batch_prediction_empty_files_rejected(client, monkeypatch):
    """files 字段缺失（空文件列表）时，FastAPI 在参数校验阶段拒绝（422），不进入推理。"""
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post("/predictions/batch")

    assert resp.status_code == 422
    fake.assert_not_called()


def test_task_create_returns_pending_then_completed(client, monkeypatch):
    """创建任务应立即返回 pending，后台预测完成后查询得到 completed 与结果。"""
    fake = mock.Mock(
        return_value=DetectionResult(
            image_width=320,
            image_height=240,
            inference_time_ms=5.0,
        )
    )
    monkeypatch.setattr("app.api.predict_image", fake)

    created = client.post(
        "/tasks",
        files={"file": ("valid.png", _png_bytes(), "image/png")},
    )

    assert created.status_code == 200
    task_id = created.json()["task_id"]
    assert created.json() == {
        "task_id": task_id,
        "status": "pending",
        "filename": "valid.png",
        "result": None,
        "error": None,
    }

    status = client.get(f"/tasks/{task_id}")

    assert status.status_code == 200
    assert status.json() == {
        "task_id": task_id,
        "status": "completed",
        "filename": "valid.png",
        "result": {
            "image_width": 320,
            "image_height": 240,
            "inference_time_ms": 5.0,
            "detections": [],
        },
        "error": None,
    }
    # 后台任务执行完成后临时文件应已清理
    assert fake.call_count == 1
    assert not fake.call_args.args[0].exists()


@pytest.mark.parametrize(
    ("exc", "error_code", "message"),
    [
        (InvalidImageError("图片无法解码"), "INVALID_IMAGE", "图片无法解码"),
        (ModelNotFoundError("模型文件不存在"), "MODEL_NOT_FOUND", "模型文件不存在"),
    ],
)
def test_task_failed_status(client, monkeypatch, exc, error_code, message):
    """后台预测失败时，任务状态应变为 failed 并携带对应错误信息。"""
    fake = mock.Mock(side_effect=exc)
    monkeypatch.setattr("app.api.predict_image", fake)

    created = client.post(
        "/tasks",
        files={"file": ("broken.png", _png_bytes(), "image/png")},
    )
    task_id = created.json()["task_id"]

    status = client.get(f"/tasks/{task_id}")

    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["result"] is None
    assert status.json()["error"] == {
        "error_code": error_code,
        "message": message,
    }


def test_task_not_found(client):
    """查询不存在的任务应返回 404 TASK_NOT_FOUND。"""

    resp = client.get("/tasks/does-not-exist")

    _assert_error_response(resp, 404, "TASK_NOT_FOUND", "任务不存在: does-not-exist")


def test_task_rejects_wrong_mime_type(client, monkeypatch):
    """创建任务时上传非图片 MIME 类型应返回 400，且不会创建后台任务。"""
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/tasks",
        files={"file": ("valid.png", _png_bytes(), "text/plain")},
    )

    _assert_error_response(resp, 400, "INVALID_IMAGE", "不支持的图片 MIME 类型: 'text/plain'")
    fake.assert_not_called()


def test_openapi_declares_task_not_found_response():
    """OpenAPI 中 GET /tasks/{task_id} 的 404 响应应引用 ErrorResponse schema。"""
    schema = app.openapi()
    responses = schema["paths"]["/tasks/{task_id}"]["get"]["responses"]

    assert "404" in responses
    assert responses["404"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/ErrorResponse"
    )


def test_prediction_rejects_oversized_file(client, monkeypatch):
    """上传超过大小限制的图片应返回 400 FILE_TOO_LARGE，且不进入推理。"""
    monkeypatch.setenv("VISION_MAX_FILE_SIZE_MB", "1")
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions",
        files={"file": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
    )

    _assert_error_response(resp, 400, "FILE_TOO_LARGE", "图片大小超过限制（最大 1 MB）")
    fake.assert_not_called()


def test_batch_prediction_rejects_oversized_file(client, monkeypatch):
    """批量上传中任一文件超过大小限制时，整个请求返回 400 FILE_TOO_LARGE。"""
    monkeypatch.setenv("VISION_MAX_FILE_SIZE_MB", "1")
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", ("large.png", b"x" * (1024 * 1024 + 1), "image/png")),
            ("files", ("small.png", _png_bytes(), "image/png")),
        ],
    )

    _assert_error_response(resp, 400, "FILE_TOO_LARGE", "图片大小超过限制（最大 1 MB）")
    fake.assert_not_called()


def test_task_rejects_oversized_file(client, monkeypatch):
    """创建任务时上传超过大小限制的图片应返回 400 FILE_TOO_LARGE，不创建后台任务。"""
    monkeypatch.setenv("VISION_MAX_FILE_SIZE_MB", "1")
    fake = mock.Mock()
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/tasks",
        files={"file": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
    )

    _assert_error_response(resp, 400, "FILE_TOO_LARGE", "图片大小超过限制（最大 1 MB）")
    fake.assert_not_called()


def test_prediction_timeout(client, monkeypatch):
    """预测超过配置的时间限制时应返回 504 TIMEOUT。"""
    monkeypatch.setenv("VISION_PREDICTION_TIMEOUT", "0.1")
    fake = mock.Mock(side_effect=lambda *args, **kwargs: time.sleep(1))
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions",
        files={"file": ("slow.png", _png_bytes(), "image/png")},
    )

    _assert_error_response(resp, 504, "TIMEOUT", "预测超时（超过 0.1 秒）")


def test_batch_prediction_timeout(client, monkeypatch):
    """批量预测中任一张图片超时，整个请求应返回 504 TIMEOUT。"""
    monkeypatch.setenv("VISION_PREDICTION_TIMEOUT", "0.1")
    fake = mock.Mock(side_effect=lambda *args, **kwargs: time.sleep(1))
    monkeypatch.setattr("app.api.predict_image", fake)

    resp = client.post(
        "/predictions/batch",
        files=[
            ("files", ("slow.png", _png_bytes(), "image/png")),
        ],
    )

    _assert_error_response(resp, 504, "TIMEOUT", "预测超时（超过 0.1 秒）")


def test_task_timeout_failed(client, monkeypatch):
    """后台预测超时时，任务状态应变为 failed 并携带 TIMEOUT 错误。"""
    monkeypatch.setenv("VISION_PREDICTION_TIMEOUT", "0.1")
    fake = mock.Mock(side_effect=lambda *args, **kwargs: time.sleep(1))
    monkeypatch.setattr("app.api.predict_image", fake)

    created = client.post(
        "/tasks",
        files={"file": ("slow.png", _png_bytes(), "image/png")},
    )
    task_id = created.json()["task_id"]

    status = client.get(f"/tasks/{task_id}")

    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["result"] is None
    assert status.json()["error"] == {
        "error_code": "TIMEOUT",
        "message": "预测超时（超过 0.1 秒）",
    }
