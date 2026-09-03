import sqlite3
from pathlib import Path

DATABASE_PATH = Path("data/vision_inspector.db")


def get_connection(autocommit: bool = False) -> sqlite3.Connection:
    """获取 SQLite 数据库连接。

    autocommit=False（默认）：写语句前会隐式开启事务，需要调用方
    显式 COMMIT/ROLLBACK（通常交给 transaction() 管理）；
    autocommit=True：每条语句立即落库，适合无需组合的独立单条写入。
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row  # 以字典形式返回查询结果,默认是元组
    if autocommit:
        connection.isolation_level = None
    return connection

