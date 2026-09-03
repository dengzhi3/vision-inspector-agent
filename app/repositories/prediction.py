"""预测相关数据访问：所有 SQLite 读写集中在这里，repository 只负责执行 SQL。

写函数一律不管理事务（代码中不出现 commit/rollback）：

1. 不传 connection：使用自动提交连接（autocommit），单条语句立即落库，
   适合无需组合的独立写入；
2. 传入 connection（通常来自 app.database.session.transaction()）：
   只执行 SQL，COMMIT/ROLLBACK 由调用方或事务统一管理。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from app.database.connection import get_connection


def utc_now_iso() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _resolve_connection(
    connection: sqlite3.Connection | None,
) -> tuple[sqlite3.Connection, bool]:
    """返回 (连接, 是否由本次调用创建并负责关闭)。

    未传 connection 时打开自动提交连接；传入 connection（来自
    session.transaction()）时仅执行 SQL，事务由调用方管理。
    """
    if connection is not None:
        return connection, False
    return get_connection(autocommit=True), True


def create_model_version(
    model_name: str,
    model_path: str,
    version: str,
    connection: sqlite3.Connection | None = None,
) -> int:
    """插入一条模型版本记录，返回自增 id。"""
    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_versions (
                model_name,
                model_path,
                version,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (model_name, model_path, version, utc_now_iso()),
        )
        return int(cursor.lastrowid)
    finally:
        if owns:
            conn.close()


def create_prediction_task(
    model_version_id: int | None,
    status: str = "pending",
    connection: sqlite3.Connection | None = None,
) -> int:
    """插入一条预测任务记录，返回自增 id。"""
    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_tasks (
                status,
                model_version_id,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                status,
                model_version_id,
                utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)
    finally:
        if owns:
            conn.close()


def save_image(
    task_id: int,
    original_path: str,
    annotated_path: str | None,
    width: int,
    height: int,
    connection: sqlite3.Connection | None = None,
) -> int:
    """插入一条图片记录，返回自增 id。"""
    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO images (
                task_id,
                original_path,
                annotated_path,
                width,
                height
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                task_id,
                original_path,
                annotated_path,
                width,
                height,
            ),
        )
        return int(cursor.lastrowid)
    finally:
        if owns:
            conn.close()


def save_detection(
    task_id: int,
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: list[int],
    connection: sqlite3.Connection | None = None,
) -> int:
    """插入一条检测记录（bbox 拆为 x1/y1/x2/y2），返回自增 id。"""
    x1, y1, x2, y2 = bbox

    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO detections (
                task_id,
                class_id,
                class_name,
                confidence,
                x1,
                y1,
                x2,
                y2
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                class_id,
                class_name,
                confidence,
                x1,
                y1,
                x2,
                y2,
            ),
        )
        return int(cursor.lastrowid)
    finally:
        if owns:
            conn.close()


def save_detections(
    task_id: int,
    detections: list,
    connection: sqlite3.Connection | None = None,
) -> None:
    """逐条保存检测结果；传入 connection 时所有写入属于同一事务。"""
    for detection in detections:
        save_detection(
            task_id=task_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            connection=connection,
        )


def update_prediction_task_status(
    task_id: int,
    status: str,
    inference_time_ms: float | None = None,
    error_message: str | None = None,
    connection: sqlite3.Connection | None = None,
) -> None:
    """更新任务状态；completed/failed 时写入 completed_at。"""
    conn, owns = _resolve_connection(connection)

    completed_at = utc_now_iso() if status in {"completed", "failed"} else None

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE prediction_tasks
            SET
                status = ?,
                inference_time_ms = ?,
                completed_at = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                status,
                inference_time_ms,
                completed_at,
                error_message,
                task_id,
            ),
        )
    finally:
        if owns:
            conn.close()


def get_prediction_task(
    task_id: int,
    connection: sqlite3.Connection | None = None,
):
    """按 id 查询任务行；只读，不开启新事务。"""
    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                id,
                status,
                model_version_id,
                inference_time_ms,
                created_at,
                completed_at,
                error_message
            FROM prediction_tasks
            WHERE id = ?
            """,
            (task_id,),
        )
        return cursor.fetchone()
    finally:
        if owns:
            conn.close()


def list_predictions(
    limit: int = 100,
    connection: sqlite3.Connection | None = None,
) -> list[dict]:
    """查询历史预测任务（最新创建的在前），关联模型名与原始图片文件名。
    未保存图片的任务（例如推理失败）filename 为 None。"""
    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                pt.id,
                pt.status,
                pt.inference_time_ms,
                pt.created_at,
                pt.completed_at,
                pt.error_message,
                mv.model_name,
                im.original_path AS filename
            FROM prediction_tasks AS pt
            LEFT JOIN model_versions AS mv
                ON mv.id = pt.model_version_id
            LEFT JOIN images AS im
                ON im.task_id = pt.id
            ORDER BY pt.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        if owns:
            conn.close()


def get_prediction(
    task_id: int,
    connection: sqlite3.Connection | None = None,
) -> dict | None:
    """查询单条预测详情：任务、模型与图片信息，并附带完整检测列表；
    任务不存在时返回 None。"""
    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                pt.id,
                pt.status,
                pt.inference_time_ms,
                pt.created_at,
                pt.completed_at,
                pt.error_message,
                mv.model_name,
                mv.model_path,
                mv.version,
                im.original_path AS filename,
                im.annotated_path,
                im.width AS image_width,
                im.height AS image_height
            FROM prediction_tasks AS pt
            LEFT JOIN model_versions AS mv
                ON mv.id = pt.model_version_id
            LEFT JOIN images AS im
                ON im.task_id = pt.id
            WHERE pt.id = ?
            """,
            (task_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [description[0] for description in cursor.description]
        detail = dict(zip(columns, row))

        cursor.execute(
            """
            SELECT class_id, class_name, confidence, x1, y1, x2, y2
            FROM detections
            WHERE task_id = ?
            ORDER BY id
            """,
            (task_id,),
        )
        detail["detections"] = [
            {
                "class_id": detection[0],
                "class_name": detection[1],
                "confidence": detection[2],
                "bbox": [
                    int(detection[3]),
                    int(detection[4]),
                    int(detection[5]),
                    int(detection[6]),
                ],
            }
            for detection in cursor.fetchall()
        ]
        return detail
    finally:
        if owns:
            conn.close()


def get_or_create_model_version(
    model_name: str,
    model_path: str,
    version: str,
    connection: sqlite3.Connection | None = None,
) -> int:
    """按 (model_name, model_path, version) 查询模型版本，不存在则插入。"""
    conn, owns = _resolve_connection(connection)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id
            FROM model_versions
            WHERE model_name = ?
              AND model_path = ?
              AND version = ?
            LIMIT 1
            """,
            (
                model_name,
                model_path,
                version,
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            return int(row[0])

        cursor.execute(
            """
            INSERT INTO model_versions (
                model_name,
                model_path,
                version,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                model_name,
                model_path,
                version,
                utc_now_iso(),
            ),
        )
        return int(cursor.lastrowid)
    finally:
        if owns:
            conn.close()
