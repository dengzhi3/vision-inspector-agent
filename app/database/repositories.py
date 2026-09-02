from datetime import datetime, timezone

from app.database.connection import get_connection

def utc_now_iso() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()

def create_model_version(
        model_name: str,
        model_path: str,
        version: str,
)->int:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        '''
        INSERT INTO model_versions (
            model_name, 
            model_path, 
            version, 
            created_at
        )
        values(?,?,?,?)
        ''',
        (model_name, model_path, version, utc_now_iso()),
    )

    model_version_id = cursor.lastrowid

    connection.commit()
    connection.close()
    return int(model_version_id)

def create_prediction_task(
    model_version_id: int | None,
    status: str = "pending",
) -> int:
    connection = get_connection()
    cursor = connection.cursor()

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

    task_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return int(task_id)

def save_image(
    task_id: int,
    original_path: str,
    annotated_path: str | None,
    width: int,
    height: int,
) -> int:
    connection = get_connection()
    cursor = connection.cursor()

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

    image_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return int(image_id)

def save_detection(
    task_id: int,
    class_id: int,
    class_name: str,
    confidence: float,
    bbox: list[int],
) -> int:
    x1, y1, x2, y2 = bbox

    connection = get_connection()
    cursor = connection.cursor()

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

    detection_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return int(detection_id)

def save_detections(
    task_id: int,
    detections,
) -> None:
    for detection in detections:
        save_detection(
            task_id=task_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
        )

def update_prediction_task_status(
    task_id: int,
    status: str,
    inference_time_ms: float | None = None,
    error_message: str | None = None,
) -> None:
    connection = get_connection()
    cursor = connection.cursor()

    completed_at = None

    if status in {"completed", "failed"}:
        completed_at = utc_now_iso()

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

    connection.commit()
    connection.close()

def get_prediction_task(task_id: int):
    connection = get_connection()
    connection.row_factory = None
    cursor = connection.cursor()

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

    row = cursor.fetchone()

    connection.close()

    return row