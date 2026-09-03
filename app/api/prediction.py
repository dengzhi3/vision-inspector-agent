"""预测相关接口：单图预测 / 批量预测 / 异步任务。"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from app.api.dependencies import error_response, get_settings
from app.core.config import MAX_BATCH_FILES, Settings
from app.schemas.prediction import (
    BatchItem,
    BatchResponse,
    ErrorResponse,
    PredictionHistory,
    PredictionResponse,
    TaskStatus,
    TaskStatusResponse,
)
from app.services.prediction import (
    PredictionTimeoutError,
    get_history,
    predict_with_timeout,
    run_batch_prediction,
    run_prediction,
)
from app.utils.file import ensure_file_size, is_supported_mime_type, save_upload_to_temp, unlink_quietly
from app.vision.model_loader import ModelNotFoundError
from app.vision.predictor import InvalidImageError

router = APIRouter()

# OpenAPI 中声明的错误响应，统一引用 ErrorResponse schema
_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    400: {
        "model": ErrorResponse,
        "description": (
            "上传的图片无效或无法解码推理（INVALID_IMAGE），"
            "批量上传超过数量上限（TOO_MANY_FILES），"
            "或上传文件超过大小限制（FILE_TOO_LARGE）"
        ),
    },
    500: {
        "model": ErrorResponse,
        "description": "模型权重文件缺失（MODEL_NOT_FOUND）",
    },
    504: {
        "model": ErrorResponse,
        "description": "预测超时（TIMEOUT）",
    },
}

# /tasks 接口的错误响应：在通用 400/500 之外增加 404
_TASK_ERROR_RESPONSES: dict[int, dict[str, object]] = {
    **_ERROR_RESPONSES,
    404: {
        "model": ErrorResponse,
        "description": "任务不存在（TASK_NOT_FOUND）",
    },
}

# 简单版本：任务状态保存在进程内存中，服务重启后任务记录丢失
_TASKS: dict[str, TaskStatusResponse] = {}


def _ensure_supported_mime_type(content_type: str | None) -> None:
    """拒绝非图片 MIME 类型的上传，抛 InvalidImageError 由统一处理器转为 400。"""
    if not is_supported_mime_type(content_type):
        raise InvalidImageError(f"不支持的图片 MIME 类型: {content_type!r}")


@router.post("/predictions", response_model=PredictionResponse, responses=_ERROR_RESPONSES)
async def create_prediction(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """上传单张图片：服务层负责落库与推理编排。"""
    _ensure_supported_mime_type(file.content_type)
    content = await file.read()
    return await run_prediction(content, file.filename or "", settings)


@router.post(
    "/predictions/batch",
    response_model=BatchResponse,
    responses=_ERROR_RESPONSES,
)
async def create_batch_prediction(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> BatchResponse:
    """一次性上传多张图片，逐张顺序预测，按上传顺序返回每个文件的结果。"""
    if len(files) > MAX_BATCH_FILES:
        return error_response(
            status_code=400,
            error_code="TOO_MANY_FILES",
            message=f"一次最多上传 {MAX_BATCH_FILES} 张图片",
        )

    uploads: list[tuple[str, bytes]] = []
    for file in files:
        _ensure_supported_mime_type(file.content_type)
        uploads.append((file.filename or "", await file.read()))

    results = await run_batch_prediction(uploads, settings)
    return BatchResponse(
        results=[
            BatchItem(filename=name, result=result)
            for (name, _content), result in zip(uploads, results)
        ],
        total=len(results),
    )


async def _run_prediction_task(task_id: str, image_path: Path, settings: Settings) -> None:
    """后台执行单张图片预测并更新任务状态，无论成败都清理临时文件。"""
    task = _TASKS[task_id]
    task.status = TaskStatus.RUNNING
    try:
        task.result = await predict_with_timeout(image_path, settings)
        task.status = TaskStatus.COMPLETED
    except PredictionTimeoutError as exc:
        task.status = TaskStatus.FAILED
        task.error = ErrorResponse(error_code="TIMEOUT", message=str(exc))
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
        unlink_quietly(image_path)


@router.get("/", response_model=list[PredictionHistory])
def history() -> list[PredictionHistory]:
    """查询数据库持久化的历史预测记录（最新创建的在前）。"""
    return get_history()


@router.post("/tasks", response_model=TaskStatusResponse, responses=_TASK_ERROR_RESPONSES)
async def create_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> TaskStatusResponse:
    """上传一张图片创建异步预测任务：立即返回 pending，后台完成后通过状态接口查询。"""
    _ensure_supported_mime_type(file.content_type)
    task_id = uuid.uuid4().hex
    filename = file.filename or ""
    task = TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        filename=filename,
    )
    _TASKS[task_id] = task

    content = await file.read()
    ensure_file_size(content, settings)
    tmp_path = save_upload_to_temp(content, Path(filename).suffix.lower())

    background_tasks.add_task(_run_prediction_task, task_id, tmp_path, settings)
    return task


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    responses=_TASK_ERROR_RESPONSES,
)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """查询任务当前状态与结果。"""
    task = _TASKS.get(task_id)
    if task is None:
        return error_response(
            status_code=404,
            error_code="TASK_NOT_FOUND",
            message=f"任务不存在: {task_id}",
        )
    return task
