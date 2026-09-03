import sqlite3

connection = sqlite3.connect('practice.db')

cursor = connection.cursor()

# cursor.execute('''
#     CREATE TABLE IF NOT EXISTS students (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         name TEXT NOT NULL,
#         age INTEGER,
#         score REAL
#     )
# ''')

# students = [
#     ("Alice", 20, 91.5),
#     ("Bob", 21, 84.0),
#     ("Charlie", 22, 76.5),
# ]

# cursor.executemany(
#     """
#     INSERT INTO students (name, age, score)
#     VALUES (?, ?, ?)
#     """,
#     students,
# )



# connection.commit()

# cursor.execute(
#     '''
#     SELECT * FROM students
#     '''
# )

# rows = cursor.fetchall()

# print(rows)

# cursor.execute(
#     '''
#     SELECT * FROM students WHERE score > ?
#     ''',
#     (80,),
# )

# rows = cursor.fetchall()

# print(rows)

# cursor.execute(
#     '''
#     SELECT * 
#     FROM students
#     WHERE name = ?
#     ''',
#     ("Alice",),
# )

# rows = cursor.fetchone()

# print(rows)

# cursor.execute(
#     '''
#     SELECT *
#     FROM students
#     ORDER BY score ASC
#     LIMIT 2
#     '''
# )

# rows = cursor.fetchall()

# print(rows)

# cursor.execute(
#     '''
#     SELECT * 
#     FROM students
#     '''
# )

# rows = cursor.fetchall()
# print(*(row[1] for row in rows)) 
#  # 打印第一行数据

# cursor.execute(
#     '''
#     SELECT name, score 
#     FROM students
#     '''
# )

# rows = cursor.fetchall()

# print(rows) 

# cursor.execute(
#     '''
#     SELECT * 
#     FROM students
#     WHERE score > ?
#     ''',
#     (80,),
# )

# rows = cursor.fetchall()

# print(rows[0])

# cursor.execute(
#     '''
#     SELECT * 
#     FROM students
#     WHERE name = ?
#     ''',
#     ("Bob",),
# )

# rows = cursor.fetchone()

# print(rows)

# cursor.execute(
#     '''
#     SELECT * 
#     FROM students
#     ORDER BY score DESC
#     '''
    
# )
# Rows = cursor.fetchall()
# print(Rows)


# cursor.execute(
#     '''
#     SELECT * 
#     FROM students
#     ORDER BY score DESC
#     limit 2
#     '''
    
# )
# Rows = cursor.fetchall()
# print(Rows)


print(sqlite3.sqlite_version)
connection.close()