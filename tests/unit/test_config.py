"""app.core.config 的单元测试。"""

from __future__ import annotations

import pytest

from app.core.config import Settings


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, 0.25),  # 未设置时使用默认值
        ("0.05", 0.05),
        ("0.5", 0.5),
        ("0.9", 0.9),
    ],
)
def test_confidence_threshold_from_env(monkeypatch, env_value, expected):
    """VISION_CONFIDENCE 环境变量应正确解析为置信度阈值。"""
    if env_value is None:
        monkeypatch.delenv("VISION_CONFIDENCE", raising=False)
    else:
        monkeypatch.setenv("VISION_CONFIDENCE", env_value)

    settings = Settings.from_env()

    assert settings.confidence_threshold == expected


@pytest.mark.parametrize(
    ("env_value", "expected_mb"),
    [
        (None, 10),  # 未设置时使用默认值 10 MB
        ("1", 1),
        ("5", 5),
        ("20", 20),
    ],
)
def test_max_file_size_from_env(monkeypatch, env_value, expected_mb):
    """VISION_MAX_FILE_SIZE_MB 环境变量应正确解析为字节大小上限。"""
    if env_value is None:
        monkeypatch.delenv("VISION_MAX_FILE_SIZE_MB", raising=False)
    else:
        monkeypatch.setenv("VISION_MAX_FILE_SIZE_MB", env_value)

    settings = Settings.from_env()

    assert settings.max_file_size_bytes == expected_mb * 1024 * 1024


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, 30.0),  # 未设置时使用默认值 30 秒
        ("5", 5.0),
        ("0.5", 0.5),
    ],
)
def test_prediction_timeout_from_env(monkeypatch, env_value, expected):
    """VISION_PREDICTION_TIMEOUT 环境变量应正确解析为预测超时（秒）。"""
    if env_value is None:
        monkeypatch.delenv("VISION_PREDICTION_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("VISION_PREDICTION_TIMEOUT", env_value)

    settings = Settings.from_env()

    assert settings.prediction_timeout_seconds == expected
