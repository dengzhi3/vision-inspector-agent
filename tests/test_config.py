"""app.config 的单元测试。"""

from __future__ import annotations

import pytest

from app.config import Settings


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
