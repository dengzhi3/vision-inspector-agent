import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.model_loader import ModelNotFoundError
from app.predictor import InvalidImageError, predict_image
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    InferenceConfig,
    ModelFileInfo,
    ModelInfoResponse,
    PredictionResponse,
)


app = FastAPI(
    title="Vision Inspector Agent",
    version="0.1.0",
    description="A FastAPI application for image inspection and analysis.",
)


def _error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    """构造符合 ErrorResponse schema 的统一错误响应。"""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error_code=error_code, message=message).model_dump(),
    )


@app.exception_handler(InvalidImageError)
async def invalid_image_error_handler(
    request: Request, exc: InvalidImageError
) -> JSONResponse:
    """图片无法解码或推理失败时返回 400，响应体使用统一错误结构。"""
    return _error_response(
        status_code=400,
        error_code="INVALID_IMAGE",
        message=str(exc),
    )


@app.exception_handler(ModelNotFoundError)
async def model_not_found_error_handler(
    request: Request, exc: ModelNotFoundError
) -> JSONResponse:
    """模型权重文件缺失时返回 500，响应体使用统一错误结构。"""
    return _error_response(
        status_code=500,
        error_code="MODEL_NOT_FOUND",
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


@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction(file: UploadFile = File(...)) -> PredictionResponse:
    """上传单张图片，临时落盘后调用 app.predictor.predict_image() 推理，返回统一 JSON 结果。"""
    settings = Settings.from_env()
    suffix = Path(file.filename or "").suffix.lower()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = await run_in_threadpool(predict_image, tmp_path, settings=settings)
        return PredictionResponse(**result.to_dict())
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
