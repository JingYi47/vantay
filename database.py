import sqlite3
import json
from datetime import datetime


DATABASE_NAME = "fingerprint.db"


def get_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            template TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def add_person(person_code, name, minutiae_points):
    conn = get_connection()

    cursor = conn.cursor()

    template = json.dumps(minutiae_points)

    cursor.execute("""
        INSERT INTO persons (
            person_code,
            name,
            template,
            created_at
        )
        VALUES (?, ?, ?, ?)
    """, (
        person_code,
        name,
        template,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def get_all_persons():
    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, person_code, name, template, created_at
        FROM persons
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    persons = []

    for row in rows:
        persons.append({
            "id": row["id"],
            "person_code": row["person_code"],
            "name": row["name"],
            "template": json.loads(row["template"]),
            "created_at": row["created_at"]
        })

    return persons


def delete_person(person_id):
    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM persons WHERE id = ?",
        (person_id,)
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_table()

    print("Database created successfully!")
    print(f"Database file: {DATABASE_NAME}")
    print()

    persons = get_all_persons()

    for person in persons:
        points = person["template"]

        print("ID:", person["id"])
        print("Mã:", person["person_code"])
        print("Tên:", person["name"])
        print("Số minutiae:", len(points))
        print("Một vài điểm đầu:", points[:3])
        print("-" * 40)
