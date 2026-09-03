"""模型加载：保证整个进程内模型只加载一次。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO


class ModelNotFoundError(FileNotFoundError):
    """模型权重文件不存在时抛出。"""


@lru_cache(maxsize=1)
def load_model(model_path: str | Path) -> YOLO:
    """加载 YOLO 模型。

    使用 lru_cache 缓存，同一个模型路径在进程内只会真正加载一次，
    后续调用直接返回缓存的模型实例。
    """
    path = Path(model_path)
    if not path.is_file():
        raise ModelNotFoundError(f"模型文件不存在: {path}")
    return YOLO(str(path))
