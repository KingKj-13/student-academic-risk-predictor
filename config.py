"""Central configuration: file locations, feature lists and shared limits."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "student_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_data.csv"
METRICS_PATH = PROCESSED_DIR / "metrics.json"
MODEL_PATH = Path(os.getenv("MODEL_PATH") or BASE_DIR / "models" / "risk_model.joblib")
DB_PATH = Path(os.getenv("DB_PATH") or BASE_DIR / "students.db")

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "attendance_pct",
    "internal_marks",
    "study_hours_weekly",
    "past_failures",
    "assignment_score",
]
CATEGORICAL_FEATURES = ["extracurricular"]
# Gender is collected and stored but deliberately NOT given to the model.
# It is used only to check whether prediction errors differ between groups.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "risk_level"
RISK_LEVELS = ["Low", "Medium", "High"]

FEATURE_LABELS = {
    "attendance_pct": "Attendance",
    "internal_marks": "Internal marks",
    "study_hours_weekly": "Weekly study hours",
    "past_failures": "Past failures",
    "assignment_score": "Assignment score",
    "extracurricular": "Extracurricular participation",
}

# Valid ranges, shared by data cleaning and request validation.
RANGES = {
    "attendance_pct": (0, 100),
    "internal_marks": (0, 100),
    "study_hours_weekly": (0, 80),
    "past_failures": (0, 20),
    "assignment_score": (0, 100),
}

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_ROWS = 5000
