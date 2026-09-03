"""FastAPI 应用入口：装配 app、注册异常处理器与系统端点。"""

from __future__ import annotations

import contextlib

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

from app.api.dependencies import error_response
from app.api.prediction import router as prediction_router
from app.core.config import Settings
from app.database.models import init_database
from app.schemas.prediction import (
    HealthResponse,
    InferenceConfig,
    ModelFileInfo,
    ModelInfoResponse,
)
from app.services.prediction import PredictionTimeoutError
from app.utils.file import FileTooLargeError
from app.vision.model_loader import ModelNotFoundError
from app.vision.predictor import InvalidImageError


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """应用启动时初始化 SQLite 表结构（幂等）。"""
    init_database()
    yield


app = FastAPI(
    title="Vision Inspector Agent",
    version="0.1.0",
    description="A FastAPI application for image inspection and analysis.",
    lifespan=lifespan,
)

app.include_router(prediction_router)


@app.exception_handler(InvalidImageError)
async def invalid_image_error_handler(
    request: Request, exc: InvalidImageError
) -> JSONResponse:
    """图片无法解码或推理失败时返回 400，响应体使用统一错误结构。"""
    return error_response(
        status_code=400,
        error_code="INVALID_IMAGE",
        message=str(exc),
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_error_handler(
    request: Request, exc: ModelNotFoundError
) -> JSONResponse:
    """模型权重文件缺失时返回 500，响应体使用统一错误结构。"""
    return error_response(
        status_code=500,
        error_code="MODEL_NOT_FOUND",
        message=str(exc),
    )


@app.exception_handler(FileTooLargeError)
async def file_too_large_error_handler(
    request: Request, exc: FileTooLargeError
) -> JSONResponse:
    """上传文件超过大小限制时返回 400，响应体使用统一错误结构。"""
    return error_response(
        status_code=400,
        error_code="FILE_TOO_LARGE",
        message=str(exc),
    )


@app.exception_handler(PredictionTimeoutError)
async def prediction_timeout_error_handler(
    request: Request, exc: PredictionTimeoutError
) -> JSONResponse:
    """预测超过时间限制时返回 504，响应体使用统一错误结构。"""
    return error_response(
        status_code=504,
        error_code="TIMEOUT",
        message=str(exc),
    )


@app.get("/")
def root():
    return {"message": "Vision Inspector API"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """返回当前模型信息，不加载模型、不执行检测，仅读取 config 配置。"""
    settings = Settings.from_env()
    model_path = settings.model_path
    return ModelInfoResponse(
        model=ModelFileInfo(
            path=str(model_path),
            exists=model_path.is_file(),
            size_bytes=model_path.stat().st_size if model_path.is_file() else None,
        ),
        inference=InferenceConfig(
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            image_size=settings.image_size,
            device=settings.device,
        ),
        output_dir=str(settings.output_dir),
    )


@app.post("/upload")
async def upload(file: UploadFile):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
    }
