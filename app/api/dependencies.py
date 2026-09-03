"""接口层共享依赖与工具。"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.schemas.prediction import ErrorResponse


def get_settings() -> Settings:
    """读取当前配置（作为 FastAPI 依赖注入）。"""
    return Settings.from_env()


def error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    """构造符合 ErrorResponse schema 的统一错误响应。"""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error_code=error_code, message=message).model_dump(),
    )
