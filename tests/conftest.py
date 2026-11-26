"""Shared fixtures: a model trained once, and a fresh database for every test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_data  # noqa: E402
import train  # noqa: E402
from main import create_app  # noqa: E402

STRONG = {
    "student_id": 1, "name": "Strong Student", "attendance_pct": 94, "internal_marks": 85,
    "study_hours_weekly": 20, "past_failures": 0, "assignment_score": 90,
    "gender": "Female", "extracurricular": "Yes",
}
WEAK = {
    "student_id": 2, "name": "Weak Student", "attendance_pct": 45, "internal_marks": 28,
    "study_hours_weekly": 3, "past_failures": 3, "assignment_score": 32,
    "gender": "Male", "extracurricular": "No",
}


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("model")
    df, summary = train.clean(generate_data.add_defects(generate_data.generate()))
    model_path = folder / "risk_model.joblib"
    metrics = train.train(df, model_path=model_path, folder=folder)
    return {"model_path": model_path, "metrics": metrics, "data": df, "cleaning": summary}


@pytest.fixture
def client(trained, tmp_path, monkeypatch):
    for variable in ("APP_USERNAME", "APP_PASSWORD", "REQUIRE_AUTH", "SMTP_HOST", "ALERT_TO"):
        monkeypatch.delenv(variable, raising=False)
    app = create_app(db_path=tmp_path / "test.db", model_path=trained["model_path"])
    with TestClient(app) as test_client:
        yield test_client
