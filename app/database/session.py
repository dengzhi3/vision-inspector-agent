"""会话/事务管理：事务边界统一在这里控制，repository 不碰 commit/rollback。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from app.database.connection import get_connection


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """开启一个数据库事务：正常退出 COMMIT，异常退出 ROLLBACK，连接始终关闭。

    用法::

        with transaction() as connection:
            create_prediction_task(..., connection=connection)
            save_image(..., connection=connection)
    """
    connection = get_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
