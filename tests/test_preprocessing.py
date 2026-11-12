"""Basic tests for data loading and preprocessing. Run with:  python -m pytest"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    clean_data,
    get_features_and_target,
    load_raw_data,
    split_data,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw_data()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_data(raw_df)


def test_raw_data_shape(raw_df):
    # Official UCI student-mat.csv: 395 students, 33 columns
    assert raw_df.shape == (395, 33)


def test_clean_gives_numeric_grades(clean_df):
    for col in ["G1", "G2", "G3"]:
        assert pd.api.types.is_numeric_dtype(clean_df[col])


def test_clean_converts_text_numbers():
    # If grades ever arrive as text (e.g. "5"), clean_data converts them to numbers
    df = load_raw_data().head(3).copy()
    df["G1"] = df["G1"].astype(str)
    assert pd.api.types.is_numeric_dtype(clean_data(df)["G1"])


def test_clean_has_no_missing_or_duplicates(clean_df):
    assert clean_df.isna().sum().sum() == 0
    assert clean_df.duplicated().sum() == 0


def test_clean_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_clean_handles_missing_and_duplicates():
    df = load_raw_data().head(5).copy()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add a duplicate
    df.loc[1, "age"] = np.nan                                # add a missing number
    df.loc[2, "Mjob"] = np.nan                               # add a missing category
    out = clean_data(df)
    assert len(out) == 5
    assert out.isna().sum().sum() == 0


def test_clean_rejects_out_of_range_values():
    df = load_raw_data().head(3).copy()
    df.loc[0, "studytime"] = 9
    with pytest.raises(ValueError):
        clean_data(df)


def test_features_exclude_g1_g2_and_target(clean_df):
    X, y = get_features_and_target(clean_df)
    assert "G1" not in X.columns and "G2" not in X.columns and "G3" not in X.columns
    assert list(X.columns) == ALL_FEATURES
    assert len(ALL_FEATURES) == 30
    assert y.name == "G3"


def test_feature_lists_cover_all_inputs(raw_df):
    expected = set(raw_df.columns) - {"G1", "G2", "G3"}
    assert set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES) == expected
    assert not set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)


def test_split_sizes_and_reproducibility(clean_df):
    X, y = get_features_and_target(clean_df)
    X_tr1, X_te1, _, _ = split_data(X, y)
    X_tr2, X_te2, _, _ = split_data(X, y)
    assert len(X_te1) == 79 and len(X_tr1) == 316
    assert list(X_te1.index) == list(X_te2.index)          # same seed -> same split
    assert not set(X_tr1.index) & set(X_te1.index)          # no overlap


def test_preprocessor_fits_on_training_only(clean_df):
    X, y = get_features_and_target(clean_df)
    X_train, X_test, _, _ = split_data(X, y)
    pre = build_preprocessor().fit(X_train)
    scaler = pre.named_transformers_["num"]
    # The scaler's learned mean must equal the TRAINING mean, not the full-data mean
    np.testing.assert_allclose(scaler.mean_, X_train[NUMERICAL_FEATURES].mean().values)
    transformed = pre.transform(X_test)
    assert transformed.shape[0] == len(X_test)
    assert not np.isnan(np.asarray(transformed.todense() if hasattr(transformed, "todense")
                                   else transformed)).any()
