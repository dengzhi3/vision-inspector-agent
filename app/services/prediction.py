"""预测业务层：编排"校验 -> 记录任务 -> 推理 -> 持久化"的完整流程。

分层约定：本层负责流程与事务边界；app.vision.predictor 只做推理，
app.repositories 只执行 SQL，app.api 只负责接收请求与返回响应。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.database.session import transaction
from app.repositories.prediction import (
    create_prediction_task,
    get_or_create_model_version,
    list_predictions,
    save_detections,
    save_image,
    update_prediction_task_status,
)
from app.schemas.prediction import PredictionHistory, PredictionResponse
from app.utils.file import ensure_file_size, save_upload_to_temp, unlink_quietly
from app.vision.predictor import predict_image


class PredictionTimeoutError(TimeoutError):
    """预测超过时间限制时抛出。"""


def _resolve_model_version(settings: Settings) -> tuple[str, str, str]:
    """从配置推导 model_versions 记录的 (model_name, model_path, version)。

    简单版本：model_name 取模型文件名（不含扩展名），version 固定为
    "current"，表示当前配置指向的模型；更换模型路径会自然生成新的版本行。
    """
    model_name = settings.model_path.stem or settings.model_path.name
    return model_name, str(settings.model_path), "current"


def _mark_task_failed(db_task_id: int | None, error: Exception) -> None:
    """尽力把任务标记为 failed；数据库本身出错时不掩盖原始异常。"""
    if db_task_id is None:
        return
    try:
        update_prediction_task_status(
            task_id=db_task_id,
            status="failed",
            error_message=str(error),
        )
    except Exception:
        pass


async def predict_with_timeout(image_path: Path, settings: Settings) -> PredictionResponse:
    """带超时执行单张图片预测，超时抛 PredictionTimeoutError。"""
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(predict_image, image_path, settings=settings),
            timeout=settings.prediction_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise PredictionTimeoutError(
            f"预测超时（超过 {settings.prediction_timeout_seconds:g} 秒）"
        ) from exc
    return PredictionResponse(**result.to_dict())


async def run_prediction(
    content: bytes,
    filename: str,
    settings: Settings | None = None,
) -> PredictionResponse:
    """上传一张图片的完整业务流：临时落盘 -> 事务 1 记录模型版本与任务
    -> 推理 -> 事务 2 持久化图片/检测/完成状态；失败时补偿标记任务 failed。
    """
    settings = settings or Settings.from_env()
    ensure_file_size(content, settings)
    tmp_path = save_upload_to_temp(content, Path(filename or "").suffix.lower())
    db_task_id: int | None = None
    try:
        model_name, model_path, version = _resolve_model_version(settings)
        # 事务 1：模型版本与任务开始记录要么一起成功，要么一起回滚
        with transaction() as connection:
            model_version_id = get_or_create_model_version(
                model_name=model_name,
                model_path=model_path,
                version=version,
                connection=connection,
            )
            db_task_id = create_prediction_task(
                model_version_id=model_version_id,
                status="running",
                connection=connection,
            )

        try:
            result = await predict_with_timeout(tmp_path, settings)
        except Exception as exc:
            # 预测失败：补偿标记任务 failed
            _mark_task_failed(db_task_id, exc)
            raise

        # 事务 2：图片、检测结果与 completed 状态要么全部落库，要么整体回滚
        try:
            with transaction() as connection:
                save_image(
                    task_id=db_task_id,
                    # 简单版本不保留上传原图，记录原始文件名作为来源标识
                    original_path=filename or "",
                    annotated_path=None,
                    width=result.image_width,
                    height=result.image_height,
                    connection=connection,
                )
                save_detections(
                    task_id=db_task_id,
                    detections=result.detections,
                    connection=connection,
                )
                update_prediction_task_status(
                    task_id=db_task_id,
                    status="completed",
                    inference_time_ms=result.inference_time_ms,
                    connection=connection,
                )
        except Exception as exc:
            # 事务 2 已回滚：补偿标记任务 failed，保证任务不会停留在 running
            _mark_task_failed(db_task_id, exc)
            raise

        return result
    finally:
        unlink_quietly(tmp_path)


async def run_batch_prediction(
    files: list[tuple[str, bytes]],
    settings: Settings | None = None,
) -> list[PredictionResponse]:
    """逐张顺序预测多张图片（不落库），按上传顺序返回结果，不使用 asyncio.gather。"""
    settings = settings or Settings.from_env()
    results: list[PredictionResponse] = []
    for filename, content in files:
        ensure_file_size(content, settings)
        tmp_path = save_upload_to_temp(content, Path(filename or "").suffix.lower())
        try:
            results.append(await predict_with_timeout(tmp_path, settings))
        finally:
            unlink_quietly(tmp_path)
    return results


def get_history(limit: int = 100) -> list[PredictionHistory]:
    """查询数据库中的历史预测记录（最新创建的在前）。
    读取不开启事务，repository 负责组装行数据。"""
    rows = list_predictions(limit=limit)
    return [PredictionHistory.model_validate(row) for row in rows]
