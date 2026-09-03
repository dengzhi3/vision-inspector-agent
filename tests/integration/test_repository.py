"""app.repositories.prediction 的测试。

单函数行为用 Mock 替换连接验证；事务版本用临时目录中的真实 SQLite 验证。
"""

from __future__ import annotations

import sqlite3
from unittest import mock

import pytest

import app.database.connection as connection_module
import app.repositories.prediction as repositories
from app.database.models import init_database
from app.database.session import transaction
from app.schemas import Detection


@pytest.fixture
def fake_db(monkeypatch):
    """把 get_connection 替换为返回 Mock 连接/游标的工厂。"""
    connection = mock.Mock()
    cursor = mock.Mock()
    connection.cursor.return_value = cursor
    monkeypatch.setattr(repositories, "get_connection", lambda *args, **kwargs: connection)
    return connection, cursor


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """把数据库指向临时目录，用真实 SQLite 验证事务行为。"""
    path = tmp_path / "vision_inspector_repo_test.db"
    monkeypatch.setattr(connection_module, "DATABASE_PATH", path)
    init_database()
    return path


def _count(db_path, table: str) -> int:
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        connection.close()


def test_get_or_create_model_version_returns_existing_id(fake_db):
    """模型版本已存在时应直接返回已有 id，不再插入。"""
    connection, cursor = fake_db
    cursor.fetchone.return_value = (5,)

    version_id = repositories.get_or_create_model_version(
        model_name="best",
        model_path="models/best.pt",
        version="current",
    )

    assert version_id == 5
    sql = cursor.execute.call_args.args[0]
    assert "SELECT id" in sql and "FROM model_versions" in sql
    cursor.execute.assert_called_once()
    connection.commit.assert_not_called()
    connection.close.assert_called_once()


def test_get_or_create_model_version_inserts_when_missing(fake_db):
    """模型版本不存在时应插入新行并返回新 id。"""
    connection, cursor = fake_db
    cursor.fetchone.return_value = None
    cursor.lastrowid = 9

    version_id = repositories.get_or_create_model_version(
        model_name="best",
        model_path="models/best.pt",
        version="1.0",
    )

    assert version_id == 9
    assert cursor.execute.call_count == 2
    insert_sql = cursor.execute.call_args_list[1].args[0]
    assert "INSERT INTO model_versions" in insert_sql
    name, path, version, created_at = cursor.execute.call_args_list[1].args[1]
    assert (name, path, version) == ("best", "models/best.pt", "1.0")
    assert isinstance(created_at, str) and created_at
    connection.commit.assert_not_called()
    connection.close.assert_called_once()


def test_create_prediction_task_returns_id(fake_db):
    """创建预测任务应返回自增 id，并关闭连接（不自行提交）。"""
    connection, cursor = fake_db
    cursor.lastrowid = 7

    task_id = repositories.create_prediction_task(
        model_version_id=2,
        status="running",
    )

    assert task_id == 7
    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO prediction_tasks" in sql
    status, model_version_id, created_at = params
    assert status == "running"
    assert model_version_id == 2
    assert isinstance(created_at, str) and created_at
    connection.commit.assert_not_called()
    connection.close.assert_called_once()


def test_save_detection_inserts_bbox_coordinates(fake_db):
    """save_detection 应把 bbox 拆成 x1/y1/x2/y2 写入 detections 表。"""
    connection, cursor = fake_db
    cursor.lastrowid = 3

    detection_id = repositories.save_detection(
        task_id=1,
        class_id=0,
        class_name="crack",
        confidence=0.9,
        bbox=[10, 20, 100, 200],
    )

    assert detection_id == 3
    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO detections" in sql
    assert params == (1, 0, "crack", 0.9, 10, 20, 100, 200)
    connection.commit.assert_not_called()
    connection.close.assert_called_once()


def test_save_detections_saves_each_detection(monkeypatch):
    """save_detections 应对列表中的每个检测框调用一次 save_detection。"""
    fake_save = mock.Mock()
    monkeypatch.setattr(repositories, "save_detection", fake_save)
    detections = [
        Detection(class_id=0, class_name="crack", confidence=0.9, bbox=[10, 20, 100, 200]),
        Detection(class_id=1, class_name="leak", confidence=0.8, bbox=[30, 40, 60, 70]),
    ]

    repositories.save_detections(task_id=5, detections=detections)

    assert fake_save.call_count == 2
    first_kwargs = fake_save.call_args_list[0].kwargs
    assert first_kwargs["task_id"] == 5
    assert first_kwargs["class_id"] == 0
    assert first_kwargs["class_name"] == "crack"
    assert first_kwargs["bbox"] == [10, 20, 100, 200]
    assert fake_save.call_args_list[1].kwargs["class_id"] == 1


@pytest.mark.parametrize(
    ("status", "inference_time_ms", "error_message", "completed_at_is_none"),
    [
        ("completed", 12.3, None, False),
        ("failed", None, "图片无法解码", False),
        ("running", None, None, True),
    ],
)
def test_update_prediction_task_status(
    fake_db,
    status,
    inference_time_ms,
    error_message,
    completed_at_is_none,
):
    """任务完成或失败时应写入 completed_at，中间状态不应写入。"""
    connection, cursor = fake_db

    repositories.update_prediction_task_status(
        task_id=42,
        status=status,
        inference_time_ms=inference_time_ms,
        error_message=error_message,
    )

    sql, params = cursor.execute.call_args.args
    assert "UPDATE prediction_tasks" in sql
    assert params[0] == status
    assert params[1] == inference_time_ms
    assert (params[2] is None) == completed_at_is_none
    if not completed_at_is_none:
        assert isinstance(params[2], str) and params[2]
    assert params[3] == error_message
    assert params[4] == 42
    connection.commit.assert_not_called()
    connection.close.assert_called_once()


def test_write_function_defers_to_shared_connection():
    """传入外部 connection 时，写函数不应自行提交或关闭。"""
    connection = mock.Mock()
    cursor = connection.cursor.return_value
    cursor.lastrowid = 4

    task_id = repositories.create_prediction_task(
        model_version_id=1,
        connection=connection,
    )

    assert task_id == 4
    connection.commit.assert_not_called()
    connection.rollback.assert_not_called()
    connection.close.assert_not_called()


def test_save_detections_forwards_connection(monkeypatch):
    """save_detections 应把外部 connection 透传给 save_detection。"""
    fake_save = mock.Mock()
    monkeypatch.setattr(repositories, "save_detection", fake_save)
    detection = Detection(
        class_id=0,
        class_name="crack",
        confidence=0.9,
        bbox=[10, 20, 100, 200],
    )

    repositories.save_detections(
        task_id=1,
        detections=[detection],
        connection=mock.sentinel.conn,
    )

    assert fake_save.call_args.kwargs["connection"] is mock.sentinel.conn


def test_transaction_commits_all_writes_on_success(db_path):
    """事务正常结束时，其中的全部写入应一次性落库。"""
    with transaction() as connection:
        task_id = repositories.create_prediction_task(
            model_version_id=None,
            status="running",
            connection=connection,
        )
        repositories.save_image(
            task_id=task_id,
            original_path="photo.png",
            annotated_path=None,
            width=320,
            height=240,
            connection=connection,
        )
        repositories.update_prediction_task_status(
            task_id=task_id,
            status="completed",
            inference_time_ms=5.0,
            connection=connection,
        )

    assert _count(db_path, "prediction_tasks") == 1
    assert _count(db_path, "images") == 1


def test_writes_invisible_until_transaction_commits(db_path):
    """事务提交前，其他连接不应看到未提交的写入。"""
    with transaction() as connection:
        repositories.create_prediction_task(
            model_version_id=None,
            status="running",
            connection=connection,
        )
        assert _count(db_path, "prediction_tasks") == 0

    assert _count(db_path, "prediction_tasks") == 1


def test_transaction_rolls_back_all_writes_on_error(db_path):
    """事务内抛错时，已执行但未提交的写入应全部回滚。"""
    with pytest.raises(RuntimeError):
        with transaction() as connection:
            task_id = repositories.create_prediction_task(
                model_version_id=None,
                status="running",
                connection=connection,
            )
            repositories.save_image(
                task_id=task_id,
                original_path="photo.png",
                annotated_path=None,
                width=320,
                height=240,
                connection=connection,
            )
            raise RuntimeError("中途失败")

    assert _count(db_path, "prediction_tasks") == 0
    assert _count(db_path, "images") == 0


def test_save_detections_atomic_inside_transaction(db_path):
    """同一事务内逐条保存检测，任一条失败应整体回滚，不留半截数据。"""
    from types import SimpleNamespace

    good = Detection(
        class_id=0,
        class_name="crack",
        confidence=0.9,
        bbox=[10, 20, 100, 200],
    )
    bad = SimpleNamespace(
        class_id=1,
        class_name="bad",
        confidence=0.5,
        bbox=[1, 2],  # bbox 无法解包成 x1/y1/x2/y2
    )

    with pytest.raises(ValueError):
        with transaction() as connection:
            task_id = repositories.create_prediction_task(
                model_version_id=None,
                status="running",
                connection=connection,
            )
            repositories.save_detections(
                task_id=task_id,
                detections=[good, bad],
                connection=connection,
            )

    assert _count(db_path, "prediction_tasks") == 0
    assert _count(db_path, "detections") == 0


def test_standalone_write_persists_immediately(db_path):
    """不传 connection 的独立写入应通过自动提交连接立即落库。"""
    repositories.create_prediction_task(
        model_version_id=None,
        status="running",
    )

    assert _count(db_path, "prediction_tasks") == 1
