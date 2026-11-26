"""Load the saved model, make predictions and explain them."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config

# A factor is reported when moving that one value to the typical Low-risk value
# reduces the risk score by at least this much.
MIN_EFFECT = 0.05

# Direction in which a value is worse than the reference: True means "lower is worse".
LOWER_IS_WORSE = {
    "attendance_pct": True,
    "internal_marks": True,
    "study_hours_weekly": True,
    "assignment_score": True,
    "past_failures": False,
}
FACTOR_TEXT = {
    "attendance_pct": "Low attendance",
    "internal_marks": "Low internal marks",
    "study_hours_weekly": "Insufficient study hours",
    "assignment_score": "Low assignment score",
    "past_failures": "Previous academic failures",
}


class ModelNotTrainedError(RuntimeError):
    pass


class RiskPredictor:
    def __init__(self, model_path: Path = config.MODEL_PATH) -> None:
        if not Path(model_path).exists():
            raise ModelNotTrainedError(f"No model found at {model_path}. Run 'python train.py' first.")
        # joblib files can execute code when loaded: only load files produced by train.py.
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.model_name: str = bundle["model_name"]
        self.features: list[str] = bundle["features"]
        self.reference: dict = bundle["reference"]
        self.classes = list(self.pipeline.classes_)

    def _probabilities(self, rows: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(rows[self.features])

    def _risk_score(self, probabilities: np.ndarray) -> np.ndarray:
        """0 = certainly Low, 1 = certainly High. Medium counts half."""
        high = probabilities[:, self.classes.index("High")]
        medium = probabilities[:, self.classes.index("Medium")]
        return high + 0.5 * medium

    def predict(self, students: list[dict]) -> list[dict]:
        if not students:
            return []
        frame = pd.DataFrame(students)
        probabilities = self._probabilities(frame)
        factors = self._contributing_factors(frame, probabilities)

        results = []
        for index, row in enumerate(probabilities):
            best = int(np.argmax(row))
            results.append(
                {
                    "risk_level": self.classes[best],
                    "confidence": round(float(row[best]), 4),
                    "probabilities": {c: round(float(row[self.classes.index(c)]), 4) for c in config.RISK_LEVELS},
                    "contributing_factors": factors[index] if self.classes[best] != "Low" else [],
                }
            )
        return results

    def _contributing_factors(self, frame: pd.DataFrame, probabilities: np.ndarray) -> list[list[str]]:
        """One-factor-at-a-time "what if" analysis against a typical Low-risk student.

        This describes how the model's score responds to each value. It is not proof
        that the value causes the student's difficulty.
        """
        base = self._risk_score(probabilities)
        effects = np.zeros((len(frame), len(LOWER_IS_WORSE)))
        names = list(LOWER_IS_WORSE)

        for column_index, feature in enumerate(names):
            reference = self.reference[feature]
            worse = frame[feature] < reference if LOWER_IS_WORSE[feature] else frame[feature] > reference
            if not worse.any():
                continue
            changed = frame.copy()
            changed[feature] = changed[feature].astype(float)
            changed.loc[worse, feature] = reference
            drop = base - self._risk_score(self._probabilities(changed))
            effects[:, column_index] = np.where(worse, drop, 0.0)

        output = []
        for row_index in range(len(frame)):
            order = np.argsort(effects[row_index])[::-1]
            chosen = [names[i] for i in order if effects[row_index, i] >= MIN_EFFECT]
            if not chosen:
                # No single change moves the score much: list what is clearly weak instead.
                chosen = self._clearly_weak(frame.iloc[row_index])
            output.append([FACTOR_TEXT[name] for name in chosen])
        return output

    def _clearly_weak(self, row: pd.Series) -> list[str]:
        weak = []
        for feature, lower_is_worse in LOWER_IS_WORSE.items():
            reference = self.reference[feature]
            if lower_is_worse and row[feature] < 0.8 * reference:
                weak.append(feature)
            elif not lower_is_worse and row[feature] > reference:
                weak.append(feature)
        return weak
