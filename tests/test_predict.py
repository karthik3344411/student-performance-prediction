"""Tests for prediction and input validation. Requires a trained model (python -m src.train_model)."""

import pytest

from src.data_preprocessing import ALL_FEATURES, get_features_and_target, clean_data, load_raw_data
from src.predict import (
    MODEL_PATH,
    ModelNotFoundError,
    grade_band,
    load_model,
    predict_grade,
    student_to_dataframe,
    validate_student,
)


@pytest.fixture(scope="module")
def sample_student():
    X, _ = get_features_and_target(clean_data(load_raw_data()))
    row = X.iloc[0].to_dict()
    # convert numpy types to plain Python types (like the app produces)
    return {k: (int(v) if not isinstance(v, str) else v) for k, v in row.items()}


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip("Model not trained yet - run: python -m src.train_model")
    return load_model()


def test_valid_student_has_no_errors(sample_student):
    assert validate_student(sample_student) == []


def test_validation_catches_bad_values(sample_student):
    bad = dict(sample_student, age=40, sex="X", studytime="two")
    errors = validate_student(bad)
    assert len(errors) == 3


def test_validation_catches_missing_feature(sample_student):
    bad = dict(sample_student)
    bad.pop("absences")
    assert any("Missing" in e for e in validate_student(bad))


def test_dataframe_has_training_columns(sample_student):
    df = student_to_dataframe(sample_student)
    assert list(df.columns) == ALL_FEATURES
    assert df.shape == (1, 30)


def test_prediction_in_valid_range(model, sample_student):
    result = predict_grade(model, sample_student)
    assert 0 <= result["predicted_grade"] <= 20
    assert isinstance(result["raw_prediction"], float)
    assert result["was_clipped"] == (result["raw_prediction"] != result["predicted_grade"])


def test_prediction_rejects_invalid_input(model, sample_student):
    with pytest.raises(ValueError):
        predict_grade(model, dict(sample_student, failures=10))


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")


def test_model_has_no_g1_g2(model):
    used = list(model.named_steps["preprocessor"].feature_names_in_)
    assert "G1" not in used and "G2" not in used


def test_grade_band():
    assert grade_band(5).startswith("Below")
    assert grade_band(10).startswith("Pass")
    assert grade_band(18).startswith("Very good")
