import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.model_loader import ModelNotFoundError
from app.predictor import InvalidImageError, predict_image
from app.schemas import (
    BatchItem,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    InferenceConfig,
    ModelFileInfo,
    ModelInfoResponse,
    PredictionResponse,
    TaskStatus,
    TaskStatusResponse,
)


app = FastAPI(
    title="Vision Inspector Agent",
    version="0.1.0",
    description="A FastAPI application for image inspection and analysis.",
)


# 与 app.predictor.SUPPORTED_IMAGE_EXTS 对应的标准图片 MIME 类型
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}

# 批量接口单次上传的图片数量上限
MAX_BATCH_FILES = 10


def _ensure_supported_mime_type(file: UploadFile) -> None:
    """拒绝非图片 MIME 类型的上传，抛 InvalidImageError 由统一异常处理器转为 400。"""
    if file.content_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise InvalidImageError(f"不支持的图片 MIME 类型: {file.content_type!r}")


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


# OpenAPI 中声明的错误响应，统一引用 ErrorResponse schema
_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    400: {
        "model": ErrorResponse,
        "description": (
            "上传的图片无效或无法解码推理（INVALID_IMAGE），"
            "或批量上传超过数量上限（TOO_MANY_FILES）"
        ),
    },
    500: {
        "model": ErrorResponse,
        "description": "模型权重文件缺失（MODEL_NOT_FOUND）",
    },
}

# 简单版本：任务状态保存在进程内存中，服务重启后任务记录丢失
_TASKS: dict[str, TaskStatusResponse] = {}

# /tasks 接口的错误响应：在通用 400/500 之外增加 404
_TASK_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    **_ERROR_RESPONSES,
    404: {
        "model": ErrorResponse,
        "description": "任务不存在（TASK_NOT_FOUND）",
    },
}


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


@app.post("/predictions", response_model=PredictionResponse, responses=_ERROR_RESPONSES)
async def create_prediction(file: UploadFile = File(...)) -> PredictionResponse:
    """上传单张图片，临时落盘后调用 app.predictor.predict_image() 推理，返回统一 JSON 结果。"""
    settings = Settings.from_env()
    _ensure_supported_mime_type(file)
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


@app.post(
    "/predictions/batch",
    response_model=BatchResponse,
    responses=_ERROR_RESPONSES,
)
async def create_batch_prediction(files: list[UploadFile] = File(...)) -> BatchResponse:
    """一次性上传多张图片，逐张顺序预测，按上传顺序返回每个文件的结果。

    不使用 asyncio.gather：每张图片依次推理，任一张失败即按单图错误规则
    抛出异常（400 INVALID_IMAGE / 500 MODEL_NOT_FOUND），整个请求失败。
    """
    if len(files) > MAX_BATCH_FILES:
        return _error_response(
            status_code=400,
            error_code="TOO_MANY_FILES",
            message=f"一次最多上传 {MAX_BATCH_FILES} 张图片",
        )
    settings = Settings.from_env()
    results: list[BatchItem] = []
    for file in files:
        _ensure_supported_mime_type(file)
        suffix = Path(file.filename or "").suffix.lower()
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = Path(tmp.name)

            result = await run_in_threadpool(predict_image, tmp_path, settings=settings)
            results.append(
                BatchItem(
                    filename=file.filename or "",
                    result=PredictionResponse(**result.to_dict()),
                )
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return BatchResponse(results=results, total=len(results))


def _run_prediction_task(task_id: str, image_path: Path, settings: Settings) -> None:
    """后台执行单张图片预测并更新任务状态，无论成败都清理临时文件。"""
    task = _TASKS[task_id]
    task.status = TaskStatus.RUNNING
    try:
        result = predict_image(image_path, settings=settings)
        task.result = PredictionResponse(**result.to_dict())
        task.status = TaskStatus.COMPLETED
    except InvalidImageError as exc:
        task.status = TaskStatus.FAILED
        task.error = ErrorResponse(error_code="INVALID_IMAGE", message=str(exc))
    except ModelNotFoundError as exc:
        task.status = TaskStatus.FAILED
        task.error = ErrorResponse(error_code="MODEL_NOT_FOUND", message=str(exc))
    except Exception as exc:  # 兜底：保证任务不会停留在 running
        task.status = TaskStatus.FAILED
        task.error = ErrorResponse(error_code="INTERNAL_ERROR", message=str(exc))
    finally:
        image_path.unlink(missing_ok=True)


@app.post("/tasks", response_model=TaskStatusResponse, responses=_TASK_ERROR_RESPONSES)
async def create_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> TaskStatusResponse:
    """上传一张图片创建异步预测任务：立即返回 pending，后台完成后通过状态接口查询。"""
    _ensure_supported_mime_type(file)
    settings = Settings.from_env()
    task_id = uuid.uuid4().hex
    filename = file.filename or ""
    task = TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        filename=filename,
    )
    _TASKS[task_id] = task

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    background_tasks.add_task(_run_prediction_task, task_id, tmp_path, settings)
    return task


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, responses=_TASK_ERROR_RESPONSES)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """查询任务当前状态与结果。"""
    task = _TASKS.get(task_id)
    if task is None:
        return _error_response(
            status_code=404,
            error_code="TASK_NOT_FOUND",
            message=f"任务不存在: {task_id}",
        )
    return task
