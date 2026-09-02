import sqlite3
from pathlib import Path

DATABASE_PATH = Path("data/vision_inspector.db")

def get_connection() -> sqlite3.Connection:
    """获取 SQLite 数据库连接。"""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

