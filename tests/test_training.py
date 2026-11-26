import numpy as np

import config
import generate_data
import train


def test_dataset_has_500_students_and_all_columns():
    data = generate_data.generate()
    assert len(data) == 500
    assert data["student_id"].is_unique
    assert set(data["risk_level"]) == {"Low", "Medium", "High"}
    assert list(data.columns) == [
        "student_id", "name", "attendance_pct", "internal_marks", "study_hours_weekly",
        "past_failures", "assignment_score", "gender", "extracurricular", "risk_level",
    ]


def test_generation_is_repeatable():
    assert generate_data.generate().equals(generate_data.generate())


def test_cleaning_removes_duplicates_and_impossible_values(trained):
    data, summary = trained["data"], trained["cleaning"]
    assert summary["duplicate_rows_removed"] == 8
    assert data["student_id"].is_unique
    for column, (low, high) in config.RANGES.items():
        assert data[column].dropna().between(low, high).all()
    assert set(data["gender"].dropna()) <= {"Male", "Female", "Other"}
    assert set(data["extracurricular"].dropna()) <= {"Yes", "No"}


def test_gender_is_not_a_model_input():
    assert "gender" not in config.FEATURES


def test_three_models_are_compared_and_one_saved(trained):
    metrics = trained["metrics"]
    assert set(metrics["models"]) == {"Logistic Regression", "Decision Tree", "Random Forest"}
    assert metrics["best_model"] in metrics["models"]
    assert trained["model_path"].exists()


def test_selected_model_beats_guessing(trained):
    metrics = trained["metrics"]
    best = metrics["models"][metrics["best_model"]]["test"]
    assert best["accuracy"] > metrics["baseline_most_frequent"]["accuracy"] + 0.15
    assert best["high_recall"] >= 0.75


def test_cleaning_keeps_missing_values_for_the_pipeline():
    data = generate_data.add_defects(generate_data.generate())
    cleaned, _ = train.clean(data)
    assert np.isnan(cleaned["attendance_pct"]).any()
