"""数据库表结构定义与初始化。"""

from __future__ import annotations

from app.database.connection import get_connection


def init_database() -> None:
    """初始化数据库，创建所需的表（幂等）。"""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            version TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            model_version_id INTEGER,
            inference_time_ms REAL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            error_message TEXT,
            FOREIGN KEY (model_version_id)
                REFERENCES model_versions(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL UNIQUE,
            original_path TEXT NOT NULL,
            annotated_path TEXT,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            FOREIGN KEY (task_id)
                REFERENCES prediction_tasks(id)
                ON DELETE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            x1 REAL NOT NULL,
            y1 REAL NOT NULL,
            x2 REAL NOT NULL,
            y2 REAL NOT NULL,
            FOREIGN KEY (task_id)
                REFERENCES prediction_tasks(id)
                ON DELETE CASCADE
        )
        """
    )

    connection.commit()
    connection.close()
