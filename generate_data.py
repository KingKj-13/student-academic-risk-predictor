"""Generate the synthetic student dataset used for development and demonstration.

The data is artificial. It is built so that the academic attributes are related
to each other and to the risk label in a plausible way, with random noise so that
no model can classify it perfectly. A few defects (missing values, duplicate rows,
inconsistent spellings, impossible values) are added on purpose so that the
preprocessing stage has real work to do.

Run:  python generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

FIRST_NAMES = [
    "Aarav", "Aditi", "Akash", "Ananya", "Arjun", "Bhavna", "Chirag", "Deepak",
    "Divya", "Farhan", "Gauri", "Harsh", "Isha", "Jatin", "Kavya", "Kunal",
    "Lakshya", "Meera", "Mohit", "Neha", "Nikhil", "Pooja", "Pranav", "Priya",
    "Rahul", "Riya", "Rohan", "Sakshi", "Sameer", "Sanya", "Shivam", "Sneha",
    "Tanvi", "Uday", "Varun", "Vidya", "Yash", "Zoya",
]
LAST_NAMES = [
    "Agarwal", "Bansal", "Chauhan", "Das", "Gupta", "Iyer", "Jain", "Kapoor",
    "Khan", "Malik", "Mehta", "Nair", "Pandey", "Patel", "Rao", "Reddy",
    "Saxena", "Sharma", "Singh", "Verma", "Yadav",
]


def generate(n_students: int = 500, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Return a clean synthetic dataset with one row per student."""
    rng = np.random.default_rng(seed)

    # Two hidden traits drive the observable attributes.
    ability = rng.normal(0, 1, n_students)
    engagement = 0.6 * ability + 0.8 * rng.normal(0, 1, n_students)

    attendance = np.clip(76 + 11 * engagement + rng.normal(0, 6, n_students), 30, 100)
    internal = np.clip(63 + 13 * ability + 4 * engagement + rng.normal(0, 7, n_students), 10, 100)
    assignment = np.clip(68 + 10 * ability + 7 * engagement + rng.normal(0, 8, n_students), 10, 100)
    study = np.clip(13 + 4 * engagement + 2 * ability + rng.normal(0, 3, n_students), 0, 40)
    failures = np.clip(rng.poisson(np.exp(-1.3 - 0.9 * ability)), 0, 5)

    # A higher score means more academic difficulty.
    score = (
        0.30 * (75 - attendance) / 10
        + 0.35 * (60 - internal) / 10
        + 0.20 * (65 - assignment) / 10
        + 0.10 * (12 - study) / 4
        + 0.55 * failures
        + rng.normal(0, 0.35, n_students)
    )
    high_cut, medium_cut = np.quantile(score, [0.82, 0.55])
    risk = np.where(score >= high_cut, "High", np.where(score >= medium_cut, "Medium", "Low"))

    names = [f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n_students)]

    return pd.DataFrame(
        {
            "student_id": np.arange(1001, 1001 + n_students),
            "name": names,
            "attendance_pct": attendance.round(1),
            "internal_marks": internal.round(0),
            "study_hours_weekly": study.round(1),
            "past_failures": failures,
            "assignment_score": assignment.round(0),
            "gender": rng.choice(["Male", "Female", "Other"], n_students, p=[0.52, 0.46, 0.02]),
            "extracurricular": rng.choice(["Yes", "No"], n_students, p=[0.45, 0.55]),
            "risk_level": risk,
        }
    )


def add_defects(df: pd.DataFrame, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Introduce the kinds of problems found in real institutional exports."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    # Missing values.
    for column in ["attendance_pct", "internal_marks", "study_hours_weekly", "assignment_score"]:
        df.loc[rng.choice(n, size=int(n * 0.02), replace=False), column] = np.nan
    df.loc[rng.choice(n, size=int(n * 0.02), replace=False), "extracurricular"] = np.nan

    # Inconsistent spellings.
    for index, value in zip(rng.choice(n, size=12, replace=False), ["male", " Female", "M", "F"] * 3):
        df.loc[index, "gender"] = value
    for index, value in zip(rng.choice(n, size=8, replace=False), ["yes ", "NO", "Y", "n"] * 2):
        df.loc[index, "extracurricular"] = value

    # Impossible values.
    bad = rng.choice(n, size=6, replace=False)
    df.loc[bad[:3], "attendance_pct"] = [104.0, 118.5, -5.0]
    df.loc[bad[3:], "study_hours_weekly"] = [-2.0, 140.0, 99.0]

    # Duplicate rows.
    duplicates = df.iloc[rng.choice(n, size=8, replace=False)]
    return pd.concat([df, duplicates], ignore_index=True)


def main(path: Path = config.RAW_DATA_PATH) -> None:
    data = add_defects(generate())
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Wrote {len(data)} rows to {path}")


if __name__ == "__main__":
    main()
