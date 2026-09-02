from app.database.connection import get_connection


def main() -> None:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    )

    tables = cursor.fetchall()

    print("Tables:")

    for table in tables:
        print(table[0])

    connection.close()


if __name__ == "__main__":
    main()