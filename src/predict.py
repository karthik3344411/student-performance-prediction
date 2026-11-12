"""
predict.py
----------
Load the saved pipeline and predict the final grade (G3) for ONE student.

Used by the Streamlit app (app.py) and by the tests.

Example:
    from src.predict import load_model, predict_grade
    model = load_model()
    result = predict_grade(model, student_dict)
    print(result["predicted_grade"])
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    VALID_CATEGORIES,
    VALID_NUMERIC_RANGES,
)

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
GRADE_MIN, GRADE_MAX = 0, 20


class ModelNotFoundError(FileNotFoundError):
    pass


def load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Model file not found: {path}. Train it first with: python -m src.train_model"
        )
    return joblib.load(path)


def load_metrics(path=METRICS_PATH):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def validate_student(student):
    """
    Check one student's input dictionary. Returns a list of error messages
    (an empty list means the input is valid).
    """
    errors = []
    missing = [f for f in ALL_FEATURES if f not in student]
    if missing:
        errors.append(f"Missing features: {', '.join(missing)}")

    for col in NUMERICAL_FEATURES:
        if col not in student:
            continue
        value = student[col]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{col} must be a number (got {value!r})")
            continue
        low, high = VALID_NUMERIC_RANGES[col]
        if not low <= value <= high:
            errors.append(f"{col} must be between {low} and {high} (got {value})")
        elif float(value) != int(value):
            errors.append(f"{col} must be a whole number (got {value})")

    for col in CATEGORICAL_FEATURES:
        if col in student and student[col] not in VALID_CATEGORIES[col]:
            errors.append(f"{col} must be one of {VALID_CATEGORIES[col]} (got {student[col]!r})")
    return errors


def student_to_dataframe(student):
    """Turn one dictionary into a 1-row DataFrame with the exact training column order."""
    row = {col: student[col] for col in ALL_FEATURES}
    df = pd.DataFrame([row], columns=ALL_FEATURES)
    for col in NUMERICAL_FEATURES:
        df[col] = df[col].astype(int)
    return df


def predict_grade(model, student):
    """
    Predict G3 for one student.

    Returns a dictionary:
      raw_prediction  - exactly what the model returned
      predicted_grade - the same value limited to the valid 0-20 grade range
      was_clipped     - True if the raw value was outside 0-20 (shown to the user,
                        never hidden)
    """
    errors = validate_student(student)
    if errors:
        raise ValueError("Invalid input: " + "; ".join(errors))

    raw = float(model.predict(student_to_dataframe(student))[0])
    clipped = min(max(raw, GRADE_MIN), GRADE_MAX)
    return {
        "raw_prediction": raw,
        "predicted_grade": clipped,
        "was_clipped": clipped != raw,
    }


def grade_band(grade):
    """Simple, descriptive band for a 0-20 Portuguese-scale grade (G3 >= 10 is a pass)."""
    if grade < 10:
        return "Below pass mark (< 10)"
    if grade < 14:
        return "Pass (10-13)"
    if grade < 16:
        return "Good (14-15)"
    return "Very good / excellent (16-20)"
