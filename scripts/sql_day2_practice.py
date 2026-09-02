import sqlite3

DATABASE_PATH = "practice_day2.db"

def create_tables() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
        '''
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class_id INTEGER,
            FOREIGN KEY (class_id)
                REFERENCES classes(id)
        )
        """
    )

    connection.commit()
    connection.close()

def insert_data() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    cursor = connection.cursor()

    cursor.executemany(
        """
        INSERT INTO classes (name)
        VALUES (?)
        """,
        [
            ("Python",),
            ("Computer Vision",),
        ],
    )

    cursor.executemany(
        """
        INSERT INTO students (name, class_id)
        VALUES (?, ?)
        """,
        [
            ("Alice", 1),
            ("Bob", 1),
            ("Charlie", 2),
        ],
    )

    connection.commit()
    connection.close()


def update_student() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE students
        SET class_id = ?
        WHERE name = ?
        """,
        (2, "Bob"),
    )

    connection.commit()
    connection.close()

def delete_student() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM students
        WHERE name = ?
        """,
        ("Charlie",),
    )

    connection.commit()
    connection.close()

def query_students_with_classes() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            students.name,
            classes.name
        FROM students
        JOIN classes
            ON students.class_id = classes.id
        """
    )

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    connection.close()

if __name__ == "__main__":
    create_tables()
    insert_data()
    update_student()
    delete_student()