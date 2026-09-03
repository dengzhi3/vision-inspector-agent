"""文件相关通用工具：MIME/大小校验、临时落盘、清理。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.core.config import SUPPORTED_IMAGE_MIME_TYPES, Settings


class FileTooLargeError(ValueError):
    """上传文件超过大小限制时抛出。"""


def is_supported_mime_type(content_type: str | None) -> bool:
    """判断 MIME 类型是否为允许上传的图片类型。"""
    return content_type in SUPPORTED_IMAGE_MIME_TYPES


def ensure_file_size(content: bytes, settings: Settings) -> None:
    """内容超过配置大小限制时抛 FileTooLargeError。"""
    if len(content) > settings.max_file_size_bytes:
        max_mb = settings.max_file_size_bytes // (1024 * 1024)
        raise FileTooLargeError(f"图片大小超过限制（最大 {max_mb} MB）")


def save_upload_to_temp(content: bytes, suffix: str) -> Path:
    """把上传内容写入临时文件并返回路径，调用方负责清理。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)


def unlink_quietly(path: Path) -> None:
    """尽力清理临时文件；被仍在运行的线程占用时忽略，交由操作系统清理。"""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
