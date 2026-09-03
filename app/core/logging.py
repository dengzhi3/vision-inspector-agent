"""日志配置：统一 logging 初始化入口。"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """配置全局日志输出格式与级别。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """获取带模块名的 logger。"""
    return logging.getLogger(name)
