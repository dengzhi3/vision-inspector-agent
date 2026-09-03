"""app.database 初始化相关测试：在临时目录中创建独立的 SQLite 库。"""

from __future__ import annotations

import sqlite3

import pytest

import app.database.connection as connection_module
from app.database.connection import get_connection
from app.database.models import init_database

EXPECTED_TABLES = {"model_versions", "prediction_tasks", "images", "detections"}


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """把数据库文件指到临时目录，保证测试不触碰真实的 data 库。"""
    path = tmp_path / "vision_inspector_test.db"
    monkeypatch.setattr(connection_module, "DATABASE_PATH", path)
    return path


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    return {row[0] for row in rows}


def test_init_database_creates_tables_in_tmp_path(db_path):
    """init_database 应在临时目录创建全部预期表。"""
    init_database()

    assert db_path.exists()
    connection = sqlite3.connect(db_path)
    try:
        assert EXPECTED_TABLES <= _table_names(connection)
    finally:
        connection.close()


def test_init_database_is_idempotent(db_path):
    """重复调用 init_database 不应报错，表结构保持不变。"""
    init_database()
    init_database()

    connection = sqlite3.connect(db_path)
    try:
        assert EXPECTED_TABLES <= _table_names(connection)
    finally:
        connection.close()


def test_connection_enables_foreign_keys(db_path):
    """get_connection 应默认开启 SQLite 外键约束。"""
    connection = get_connection()
    try:
        pragma = connection.execute("PRAGMA foreign_keys").fetchone()
        assert pragma[0] == 1
    finally:
        connection.close()
