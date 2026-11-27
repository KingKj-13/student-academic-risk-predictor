"""Clean the data, explore it, train three classifiers and save the best one.

Run:  python train.py

Outputs
    data/processed/cleaned_data.csv   cleaned dataset
    data/processed/*.png              exploratory and evaluation graphs
    data/processed/metrics.json       every number quoted in the documentation
    models/risk_model.joblib          preprocessing pipeline + model + reference values
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

import config  # noqa: E402
import generate_data  # noqa: E402

GENDER_MAP = {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"}
YES_NO_MAP = {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}
COLOURS = {"Low": "#2e7d5b", "Medium": "#c98a1b", "High": "#b3362f"}


# --------------------------------------------------------------------------- cleaning
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned copy of the data and a summary of what was changed.

    Missing values are left in place on purpose: they are imputed inside the model
    pipeline, using statistics learnt from the training split only.
    """
    df = raw.copy()
    summary = {"raw_rows": int(len(df))}

    for column, mapping in (("gender", GENDER_MAP), ("extracurricular", YES_NO_MAP)):
        normalised = df[column].astype("string").str.strip().str.lower().map(mapping)
        changed = df[column].notna() & (df[column].astype("string") != normalised.astype("string"))
        summary[f"{column}_values_normalised"] = int(changed.fillna(True).sum())
        df[column] = normalised.astype(object).where(normalised.notna(), np.nan)

    out_of_range = 0
    for column, (low, high) in config.RANGES.items():
        df[column] = pd.to_numeric(df[column], errors="coerce")
        invalid = df[column].notna() & ~df[column].between(low, high)
        out_of_range += int(invalid.sum())
        df.loc[invalid, column] = np.nan
    summary["impossible_values_removed"] = out_of_range

    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="student_id", keep="first")
    summary["duplicate_rows_removed"] = int(before - len(df))

    before = len(df)
    df = df[df[config.TARGET].isin(config.RISK_LEVELS)].reset_index(drop=True)
    summary["rows_without_valid_label_removed"] = int(before - len(df))

    summary["clean_rows"] = int(len(df))
    summary["missing_values_by_column"] = {
        column: int(count) for column, count in df[config.FEATURES].isna().sum().items()
    }
    summary["class_counts"] = {level: int((df[config.TARGET] == level).sum()) for level in config.RISK_LEVELS}
    return df, summary


# --------------------------------------------------------------------------- exploration
def _save(fig: plt.Figure, folder: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(folder / name, dpi=160)
    plt.close(fig)


def explore(df: pd.DataFrame, folder: Path) -> dict:
    """Draw the exploratory graphs and return summary statistics."""
    levels = config.RISK_LEVELS
    colours = [COLOURS[level] for level in levels]

    for column, name, title in (
        ("attendance_pct", "eda_attendance.png", "Distribution of attendance"),
        ("internal_marks", "eda_internal_marks.png", "Distribution of internal marks"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[column].dropna(), bins=25, color="#3b5b92", edgecolor="white")
        ax.set(title=title, xlabel=config.FEATURE_LABELS[column], ylabel="Students")
        _save(fig, folder, name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for level in levels:
        subset = df[df[config.TARGET] == level]
        ax.scatter(subset["study_hours_weekly"], subset["internal_marks"], s=16, alpha=0.7,
                   color=COLOURS[level], label=level)
    ax.set(title="Study hours against internal marks", xlabel="Weekly study hours", ylabel="Internal marks")
    ax.legend(title="Risk")
    _save(fig, folder, "eda_study_vs_marks.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    boxes = ax.boxplot([df.loc[df[config.TARGET] == lv, "assignment_score"].dropna() for lv in levels],
                       tick_labels=levels, patch_artist=True)
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    ax.set(title="Assignment score by risk category", xlabel="Risk category", ylabel="Assignment score")
    _save(fig, folder, "eda_assignment_by_risk.png")

    corr = df[config.NUMERIC_FEATURES].corr()
    labels = [config.FEATURE_LABELS[c] for c in config.NUMERIC_FEATURES]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.55 else "black")
    ax.set_title("Correlation between numerical attributes")
    fig.colorbar(image, ax=ax, shrink=0.8)
    _save(fig, folder, "eda_correlation.png")

    counts = [int((df[config.TARGET] == lv).sum()) for lv in levels]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(levels, counts, color=colours)
    ax.bar_label(bars)
    ax.set(title="Students in each risk category", ylabel="Students")
    _save(fig, folder, "eda_class_counts.png")

    means = df.groupby(config.TARGET)[config.NUMERIC_FEATURES].mean().loc[levels].round(2)
    return {
        "means_by_risk": means.to_dict(orient="index"),
        "describe": df[config.NUMERIC_FEATURES].describe().round(2).to_dict(),
        "correlation": corr.round(3).to_dict(),
    }


# --------------------------------------------------------------------------- modelling
def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, config.NUMERIC_FEATURES), ("categorical", categorical, config.CATEGORICAL_FEATURES)]
    )


def candidate_models() -> dict[str, tuple[object, dict]]:
    seed = config.RANDOM_STATE
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=seed),
            {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [3, 8, 15]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1),
            {"model__n_estimators": [200], "model__max_depth": [4, 6, 10, None],
             "model__min_samples_leaf": [1, 3, 8]},
        ),
    }


def selection_score(y_true, y_pred) -> float:
    """Equal weight to High-risk recall and to balanced performance over all classes."""
    high_recall = recall_score(y_true, y_pred, labels=["High"], average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=config.RISK_LEVELS, average="macro", zero_division=0)
    return 0.5 * high_recall + 0.5 * macro_f1


def evaluate(y_true, y_pred) -> dict:
    kwargs = {"labels": config.RISK_LEVELS, "zero_division": 0}
    high = {"labels": ["High"], "average": "macro", "zero_division": 0}
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", **kwargs),
        "recall_macro": recall_score(y_true, y_pred, average="macro", **kwargs),
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kwargs),
        "high_precision": precision_score(y_true, y_pred, **high),
        "high_recall": recall_score(y_true, y_pred, **high),
        "high_f1": f1_score(y_true, y_pred, **high),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=config.RISK_LEVELS).tolist(),
    }


def calibration_report(pipeline: Pipeline, x_test, y_test, folder: Path) -> dict:
    """Compare predicted probabilities with observed frequencies."""
    probabilities = pipeline.predict_proba(x_test)
    classes = list(pipeline.classes_)
    one_hot = np.array([[1.0 if label == c else 0.0 for c in classes] for label in y_test])
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    confidence = probabilities.max(axis=1)
    correct = (np.array(classes)[probabilities.argmax(axis=1)] == np.asarray(y_test)).astype(float)
    edges = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    bins, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        if not mask.any():
            continue
        mean_conf, acc, count = float(confidence[mask].mean()), float(correct[mask].mean()), int(mask.sum())
        bins.append({"range": f"{low:.1f}-{min(high, 1.0):.1f}", "count": count,
                     "mean_confidence": mean_conf, "accuracy": acc})
        ece += count / len(confidence) * abs(mean_conf - acc)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins], "o-",
            color="#3b5b92", label="Model")
    ax.set(title="Reliability of the confidence value", xlabel="Mean confidence", ylabel="Observed accuracy",
           xlim=(0.3, 1.02), ylim=(0.3, 1.02))
    ax.legend()
    _save(fig, folder, "eval_calibration.png")
    return {"brier_score": brier, "expected_calibration_error": float(ece), "bins": bins}


def fairness_report(pipeline: Pipeline, df: pd.DataFrame) -> dict:
    """High-risk recall per gender, from out-of-fold predictions over all students."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    predicted = cross_val_predict(clone(pipeline), df[config.FEATURES], df[config.TARGET], cv=folds)
    report = {}
    for group in ["Female", "Male", "Other"]:
        mask = (df["gender"] == group).to_numpy()
        actual = df.loc[mask, config.TARGET].to_numpy()
        guess = predicted[mask]
        high = actual == "High"
        report[group] = {
            "students": int(mask.sum()),
            "actual_high": int(high.sum()),
            "high_recall": float((guess[high] == "High").mean()) if high.any() else None,
            "false_high_rate": float((guess[~high] == "High").mean()) if (~high).any() else None,
            "accuracy": float((guess == actual).mean()) if mask.any() else None,
        }
    return report


def train(df: pd.DataFrame, model_path: Path = config.MODEL_PATH, folder: Path = config.PROCESSED_DIR) -> dict:
    """Tune and compare the three algorithms, then save the best pipeline."""
    x, y = df[config.FEATURES], df[config.TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    for name, (estimator, grid) in candidate_models().items():
        search = GridSearchCV(
            Pipeline([("preprocess", make_preprocessor()), ("model", estimator)]),
            grid, scoring=make_scorer(selection_score), cv=folds, n_jobs=-1,
        )
        search.fit(x_train, y_train)
        fitted[name] = search.best_estimator_
        results[name] = {
            "cv_selection_score": float(search.best_score_),
            "cv_selection_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "test": evaluate(y_test, search.predict(x_test)),
        }

    # The choice uses cross-validation on the training data only; the test set stays unseen.
    best_name = max(results, key=lambda n: results[n]["cv_selection_score"])
    best = fitted[best_name]

    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline = evaluate(y_test, dummy.predict(x_test))

    # Graphs.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, name in zip(axes, results):
        ConfusionMatrixDisplay(np.array(results[name]["test"]["confusion_matrix"]),
                               display_labels=config.RISK_LEVELS).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    _save(fig, folder, "eval_confusion_matrices.png")

    metric_keys = ["accuracy", "f1_macro", "high_precision", "high_recall"]
    metric_names = ["Accuracy", "Macro F1", "High precision", "High recall"]
    positions = np.arange(len(metric_keys))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for offset, (name, colour) in enumerate(zip(results, ["#3b5b92", "#7a9e7e", "#c98a1b"])):
        ax.bar(positions + (offset - 1) * 0.26, [results[name]["test"][k] for k in metric_keys], 0.25,
               label=name, color=colour)
    ax.set_xticks(positions, metric_names)
    ax.set(ylim=(0, 1.18), title="Test-set performance of the three algorithms")
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="upper center", ncols=3, frameon=False)
    _save(fig, folder, "eval_model_comparison.png")

    importance = permutation_importance(best, x_test, y_test, scoring=make_scorer(selection_score),
                                        n_repeats=30, random_state=config.RANDOM_STATE)
    ranked = sorted(zip(config.FEATURES, importance.importances_mean), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([config.FEATURE_LABELS[f] for f, _ in ranked], [v for _, v in ranked], color="#3b5b92")
    ax.set(title=f"Permutation importance ({best_name})", xlabel="Drop in selection score when shuffled")
    _save(fig, folder, "eval_feature_importance.png")

    calibration = calibration_report(best, x_test, y_test, folder)
    fairness = fairness_report(best, df)

    # Typical values of Low-risk students, used to explain individual predictions.
    low = x_train[y_train == "Low"]
    reference = {c: float(low[c].median()) for c in config.NUMERIC_FEATURES}
    reference.update({c: str(low[c].mode().iloc[0]) for c in config.CATEGORICAL_FEATURES})

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_class_counts": {lv: int((y_test == lv).sum()) for lv in config.RISK_LEVELS},
        "models": results,
        "baseline_most_frequent": baseline,
        "best_model": best_name,
        "feature_importance": {f: float(v) for f, v in ranked[::-1]},
        "calibration": calibration,
        "fairness_by_gender": fairness,
        "reference_low_risk": reference,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best,
            "model_name": best_name,
            "features": config.FEATURES,
            "reference": reference,
            "sklearn_version": sklearn.__version__,
            "trained_at": metrics["trained_at"],
        },
        model_path,
    )
    return metrics


def main() -> None:
    if not config.RAW_DATA_PATH.exists():
        generate_data.main()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df, cleaning = clean(pd.read_csv(config.RAW_DATA_PATH))
    df.to_csv(config.CLEANED_DATA_PATH, index=False)
    print(f"Cleaned: {cleaning['raw_rows']} -> {cleaning['clean_rows']} rows, classes {cleaning['class_counts']}")

    exploration = explore(df, config.PROCESSED_DIR)
    metrics = train(df)
    metrics["cleaning"] = cleaning
    metrics["exploration"] = exploration
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n{'Model':<22}{'CV score':>10}{'Accuracy':>10}{'Macro F1':>10}{'High prec':>11}{'High rec':>10}")
    for name, result in metrics["models"].items():
        test = result["test"]
        print(f"{name:<22}{result['cv_selection_score']:>10.3f}{test['accuracy']:>10.3f}{test['f1_macro']:>10.3f}"
              f"{test['high_precision']:>11.3f}{test['high_recall']:>10.3f}")
    print(f"\nSelected: {metrics['best_model']}\nSaved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
