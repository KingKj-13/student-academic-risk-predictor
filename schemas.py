"""Pydantic models that validate every request and describe every response."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

GENDERS = {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"}
YES_NO = {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}


class StudentIn(BaseModel):
    """One student's academic record, as submitted by a teacher."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "student_id": 101,
                "name": "Rahul Sharma",
                "attendance_pct": 52,
                "internal_marks": 34,
                "study_hours_weekly": 5,
                "past_failures": 2,
                "assignment_score": 41,
                "gender": "Male",
                "extracurricular": "No",
            }
        },
    )

    student_id: int = Field(ge=1, le=999_999_999_999)
    name: str = Field(min_length=1, max_length=100)
    attendance_pct: float = Field(ge=0, le=100, allow_inf_nan=False)
    internal_marks: float = Field(ge=0, le=100, allow_inf_nan=False)
    study_hours_weekly: float = Field(ge=0, le=80, allow_inf_nan=False)
    past_failures: int = Field(ge=0, le=20)
    assignment_score: float = Field(ge=0, le=100, allow_inf_nan=False)
    gender: Literal["Male", "Female", "Other"]
    extracurricular: Literal["Yes", "No"]

    @field_validator("gender", mode="before")
    @classmethod
    def normalise_gender(cls, value: object) -> object:
        if isinstance(value, str):
            return GENDERS.get(value.strip().lower(), value)
        return value

    @field_validator("extracurricular", mode="before")
    @classmethod
    def normalise_yes_no(cls, value: object) -> object:
        if isinstance(value, str):
            return YES_NO.get(value.strip().lower(), value)
        return value

    @field_validator("past_failures", mode="before")
    @classmethod
    def whole_number_text(cls, value: object) -> object:
        # Spreadsheets often export 2 as "2.0".
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                return value
            if math.isfinite(number) and number.is_integer():
                return int(number)
        return value


class PredictionOut(BaseModel):
    student_id: int
    name: str
    risk_level: Literal["Low", "Medium", "High"]
    confidence: float
    probabilities: dict[str, float]
    status: str
    contributing_factors: list[str]
    recommendations: list[str]
    alert: bool


class RowError(BaseModel):
    row: int
    student_id: str
    errors: list[str]


class UploadOut(BaseModel):
    total_rows: int
    processed: int
    rejected: int
    counts: dict[str, int]
    results: list[PredictionOut]
    errors: list[RowError]
