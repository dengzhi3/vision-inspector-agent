"""app.services.prediction 的单元测试：推理与仓库依赖均用 Mock。"""

from __future__ import annotations

import asyncio
import contextlib
from unittest import mock

import pytest

import app.services.prediction as service
from app.core.config import Settings
from app.schemas.prediction import DetectionResult, PredictionResponse
from app.vision.predictor import InvalidImageError


@pytest.fixture
def fake_deps(monkeypatch):
    """Mock 掉仓库函数与推理；transaction 依次提供两个连接。"""
    names = [
        "get_or_create_model_version",
        "create_prediction_task",
        "save_image",
        "save_detections",
        "update_prediction_task_status",
    ]
    mocks = {name: mock.Mock() for name in names}
    for name, repo_mock in mocks.items():
        monkeypatch.setattr(service, name, repo_mock)

    mocks["connection_1"] = mock.MagicMock()
    mocks["connection_2"] = mock.MagicMock()
    connections = iter([mocks["connection_1"], mocks["connection_2"]])

    @contextlib.contextmanager
    def fake_transaction():
        yield next(connections)

    monkeypatch.setattr(service, "transaction", fake_transaction)

    predict = mock.Mock(
        return_value=DetectionResult(
            image_width=320,
            image_height=240,
            inference_time_ms=5.0,
        )
    )
    monkeypatch.setattr(service, "predict_image", predict)
    mocks["predict_image"] = predict
    return mocks


def test_run_prediction_persists_and_returns(fake_deps):
    """成功路径应使用事务 1 记录任务、事务 2 持久化结果并返回 PredictionResponse。"""
    mocks = fake_deps
    mocks["get_or_create_model_version"].return_value = 7
    mocks["create_prediction_task"].return_value = 42
    settings = Settings()

    result = asyncio.run(
        service.run_prediction(b"fake-image-bytes", "a.png", settings)
    )

    assert isinstance(result, PredictionResponse)
    assert result.image_width == 320
    mocks["get_or_create_model_version"].assert_called_once_with(
        model_name="best",
        model_path=str(settings.model_path),
        version="current",
        connection=mocks["connection_1"],
    )
    mocks["create_prediction_task"].assert_called_once_with(
        model_version_id=7,
        status="running",
        connection=mocks["connection_1"],
    )
    mocks["save_image"].assert_called_once_with(
        task_id=42,
        original_path="a.png",
        annotated_path=None,
        width=320,
        height=240,
        connection=mocks["connection_2"],
    )
    mocks["save_detections"].assert_called_once()
    mocks["update_prediction_task_status"].assert_called_once_with(
        task_id=42,
        status="completed",
        inference_time_ms=5.0,
        connection=mocks["connection_2"],
    )
    # 预测结束后临时文件应已清理
    tmp_path = mocks["predict_image"].call_args.args[0]
    assert not tmp_path.exists()


def test_run_prediction_marks_failed_on_predict_error(fake_deps):
    """预测失败时应补偿标记任务 failed，且不写入图片/检测。"""
    mocks = fake_deps
    mocks["create_prediction_task"].return_value = 42
    mocks["predict_image"].side_effect = InvalidImageError("图片无法解码")

    with pytest.raises(InvalidImageError):
        asyncio.run(service.run_prediction(b"x", "broken.png", Settings()))

    mocks["update_prediction_task_status"].assert_called_once_with(
        task_id=42,
        status="failed",
        error_message="图片无法解码",
    )
    mocks["save_image"].assert_not_called()
    mocks["save_detections"].assert_not_called()


def test_run_prediction_raises_timeout(fake_deps):
    """推理超过配置时限时应抛出 PredictionTimeoutError。"""
    import time

    mocks = fake_deps

    def _slow(*args, **kwargs):
        time.sleep(1)
        return DetectionResult(image_width=320, image_height=240, inference_time_ms=1.0)

    mocks["predict_image"].side_effect = _slow

    with pytest.raises(service.PredictionTimeoutError, match="预测超时"):
        asyncio.run(
            service.run_prediction(
                b"x",
                "slow.png",
                Settings(prediction_timeout_seconds=0.1),
            )
        )


def test_run_batch_prediction_sequential_order(fake_deps):
    """批量预测应按上传顺序逐张返回结果，且不落库。"""
    mocks = fake_deps
    mocks["predict_image"].side_effect = [
        DetectionResult(image_width=320, image_height=240, inference_time_ms=1.0),
        DetectionResult(image_width=640, image_height=480, inference_time_ms=2.0),
    ]

    results = asyncio.run(
        service.run_batch_prediction(
            [("a.png", b"1"), ("b.png", b"2")],
            Settings(),
        )
    )

    assert [r.image_width for r in results] == [320, 640]
    assert mocks["predict_image"].call_count == 2
    mocks["create_prediction_task"].assert_not_called()
    assert all(
        not call.args[0].exists() for call in mocks["predict_image"].call_args_list
    )
