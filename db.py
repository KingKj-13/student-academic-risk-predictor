"""SQLite storage for student records and prediction history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id          INTEGER PRIMARY KEY,
    name                TEXT    NOT NULL,
    attendance_pct      REAL    NOT NULL,
    internal_marks      REAL    NOT NULL,
    study_hours_weekly  REAL    NOT NULL,
    past_failures       INTEGER NOT NULL,
    assignment_score    REAL    NOT NULL,
    gender              TEXT    NOT NULL,
    extracurricular     TEXT    NOT NULL
);

-- Each prediction keeps a snapshot of the inputs so that trends can be shown.
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id          INTEGER NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    risk_level          TEXT    NOT NULL CHECK (risk_level IN ('Low', 'Medium', 'High')),
    confidence          REAL    NOT NULL,
    contributing_factors TEXT   NOT NULL DEFAULT '[]',
    attendance_pct      REAL    NOT NULL,
    internal_marks      REAL    NOT NULL,
    study_hours_weekly  REAL    NOT NULL,
    past_failures       INTEGER NOT NULL,
    assignment_score    REAL    NOT NULL,
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_student ON predictions (student_id, prediction_id);
"""

STUDENT_COLUMNS = [
    "student_id", "name", "attendance_pct", "internal_marks", "study_hours_weekly",
    "past_failures", "assignment_score", "gender", "extracurricular",
]

# Each student joined to their most recent prediction.
LATEST = """
FROM students s
JOIN predictions p ON p.prediction_id = (
    SELECT MAX(prediction_id) FROM predictions WHERE student_id = s.student_id
)
"""

RISK_ORDER = "CASE p.risk_level WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END"


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, roll back on error, always close."""
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save_predictions(self, records: list[dict]) -> list[str | None]:
        """Store students and predictions in one transaction.

        Returns, for each record, the student's previous risk level (or None), which
        the caller uses to decide whether an alert is new.
        """
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        columns = ", ".join(STUDENT_COLUMNS)
        placeholders = ", ".join(f":{c}" for c in STUDENT_COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in STUDENT_COLUMNS[1:])
        previous: list[str | None] = []

        with self.connect() as connection:
            for record in records:
                row = connection.execute(
                    "SELECT risk_level FROM predictions WHERE student_id = ? ORDER BY prediction_id DESC LIMIT 1",
                    (record["student_id"],),
                ).fetchone()
                previous.append(row["risk_level"] if row else None)
                connection.execute(
                    f"INSERT INTO students ({columns}) VALUES ({placeholders}) "
                    f"ON CONFLICT(student_id) DO UPDATE SET {updates}",
                    record,
                )
                connection.execute(
                    """INSERT INTO predictions (student_id, risk_level, confidence, contributing_factors,
                           attendance_pct, internal_marks, study_hours_weekly, past_failures,
                           assignment_score, created_at)
                       VALUES (:student_id, :risk_level, :confidence, :factors, :attendance_pct,
                           :internal_marks, :study_hours_weekly, :past_failures, :assignment_score, :created_at)""",
                    {**record, "factors": json.dumps(record["contributing_factors"]), "created_at": now},
                )
        return previous

    def list_students(
        self, risk: str | None = None, search: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[int, list[dict]]:
        """Latest result per student, High risk first, then by confidence."""
        clauses, params = [], []
        if risk:
            clauses.append("p.risk_level = ?")
            params.append(risk)
        if search:
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("(s.name LIKE ? ESCAPE '\\' OR CAST(s.student_id AS TEXT) LIKE ? ESCAPE '\\')")
            params += [f"%{escaped}%", f"%{escaped}%"]
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self.connect() as connection:
            total = connection.execute(f"SELECT COUNT(*) {LATEST} {where}", params).fetchone()[0]
            rows = connection.execute(
                f"SELECT s.*, p.risk_level, p.confidence, p.contributing_factors, p.created_at {LATEST} {where} "
                f"ORDER BY {RISK_ORDER}, p.confidence DESC, s.student_id LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return total, [self._decode(row) for row in rows]

    def get_student(self, student_id: int) -> dict | None:
        with self.connect() as connection:
            student = connection.execute("SELECT * FROM students WHERE student_id = ?", (student_id,)).fetchone()
            if student is None:
                return None
            history = connection.execute(
                "SELECT * FROM predictions WHERE student_id = ? ORDER BY prediction_id", (student_id,)
            ).fetchall()
        return {**dict(student), "history": [self._decode(row) for row in history]}

    def delete_student(self, student_id: int) -> bool:
        """Remove a student and their history (for correction or erasure requests)."""
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        return cursor.rowcount > 0

    def stats(self) -> dict:
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT p.risk_level AS level, COUNT(*) AS n {LATEST} GROUP BY p.risk_level"
            ).fetchall()
            averages = connection.execute(
                f"SELECT COUNT(*) AS total, AVG(s.attendance_pct) AS attendance, "
                f"AVG(s.internal_marks) AS marks, AVG(s.assignment_score) AS assignment {LATEST}"
            ).fetchone()
            by_risk = connection.execute(
                f"SELECT p.risk_level AS level, AVG(s.attendance_pct) AS attendance, "
                f"AVG(s.internal_marks) AS marks, AVG(s.assignment_score) AS assignment {LATEST} "
                f"GROUP BY p.risk_level"
            ).fetchall()

        counts = {"Low": 0, "Medium": 0, "High": 0}
        counts.update({row["level"]: row["n"] for row in rows})

        def rounded(value: float | None) -> float:
            return round(value, 1) if value is not None else 0.0

        return {
            "total_students": averages["total"],
            "risk_counts": counts,
            "average_attendance": rounded(averages["attendance"]),
            "average_internal_marks": rounded(averages["marks"]),
            "average_assignment_score": rounded(averages["assignment"]),
            "averages_by_risk": {
                row["level"]: {
                    "attendance": rounded(row["attendance"]),
                    "internal_marks": rounded(row["marks"]),
                    "assignment_score": rounded(row["assignment"]),
                }
                for row in by_risk
            },
        }

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["contributing_factors"] = json.loads(data["contributing_factors"])
        return data
