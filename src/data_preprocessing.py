"""
data_preprocessing.py
---------------------
Phase 1 + Phase 2 of the project.

What this file does (in simple words):
1. Loads the UCI "student-mat.csv" file (it uses ';' as the separator).
2. Audits the data: shape, data types, missing values, duplicates.
3. Cleans the data: fixes wrong data types, removes duplicates, checks value ranges.
4. Splits the data into features (X) and target (y = G3).
5. Builds a scikit-learn ColumnTransformer that
      - scales the numerical columns   (StandardScaler)
      - one-hot encodes the text columns (OneHotEncoder)
6. Splits the data into a training set and a test set.

Run it directly to see the data audit:
    python -m src.data_preprocessing
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Project paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-mat.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student-mat-clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42  # one fixed seed everywhere -> results are reproducible
TEST_SIZE = 0.20   # 20% of students are kept aside for the final test

TARGET = "G3"                        # final grade (0-20) -> what we predict
EXCLUDED_GRADE_COLUMNS = ["G1", "G2"]  # earlier grades -> NOT used as inputs

# Text (categorical) columns. Binary yes/no columns are also treated as categorical.
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

# Number columns. Many are ordinal scales (e.g. studytime 1-4, famrel 1-5).
NUMERICAL_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Allowed values for each column (taken from the official UCI data dictionary).
# Used to check the data and to validate user input in the app.
VALID_NUMERIC_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4), "Fedu": (0, 4),
    "traveltime": (1, 4), "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 93),
    "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
}

VALID_CATEGORIES = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "address": ["U", "R"],
    "famsize": ["LE3", "GT3"],
    "Pstatus": ["T", "A"],
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "schoolsup": ["yes", "no"], "famsup": ["yes", "no"], "paid": ["yes", "no"],
    "activities": ["yes", "no"], "nursery": ["yes", "no"], "higher": ["yes", "no"],
    "internet": ["yes", "no"], "romantic": ["yes", "no"],
}


# ---------------------------------------------------------------------------
# Phase 1 - loading and auditing
# ---------------------------------------------------------------------------
def load_raw_data(path=RAW_DATA_PATH):
    """Read the raw CSV. The UCI file uses ';' (semicolon) as separator."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from https://archive.ics.uci.edu/dataset/320/student+performance "
            "and place student-mat.csv inside data/raw/."
        )
    return pd.read_csv(path, sep=";")


def audit_data(df):
    """Return a small dictionary that describes the dataset (used in notebook/report)."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_per_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def find_outliers_iqr(series):
    """Return the values that lie outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]


# ---------------------------------------------------------------------------
# Phase 2 - cleaning
# ---------------------------------------------------------------------------
def clean_data(df):
    """
    Clean the raw data and return a NEW DataFrame (the input is not modified).

    Steps:
    - Make sure all number columns really are numbers. (G1/G2 are written in quotes
      in the raw file, e.g. "5"; pandas already reads them as integers, but we
      convert explicitly so the code is safe if the file format changes.)
    - Strip extra spaces from text columns.
    - Fill missing values if any exist (median for numbers, mode for text).
      (The official file has no missing values, but the code handles it safely.)
    - Remove exact duplicate rows.
    - Check that every value is inside its documented valid range.

    Outliers (for example very high absences) are NOT removed: they are real,
    possible values, and removing them would throw away real students.
    """
    df = df.copy()

    # 1. Data type correction
    for col in ["G1", "G2", "G3"] + NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 2. Missing values (none in the official data, handled just in case)
    for col in df.columns:
        if df[col].isna().any():
            if col == TARGET:
                continue  # never invent a target value
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=[TARGET])  # rows without a target cannot be used

    # 3. Duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Range checks
    for col, (low, high) in VALID_NUMERIC_RANGES.items():
        if col in df.columns:
            bad = df[(df[col] < low) | (df[col] > high)]
            if len(bad) > 0:
                raise ValueError(f"Column {col} has {len(bad)} values outside {low}-{high}")

    df[TARGET] = df[TARGET].astype(int)
    return df


def get_features_and_target(df):
    """Split the clean DataFrame into X (inputs) and y (target G3). G1/G2 are excluded."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def build_preprocessor():
    """
    ColumnTransformer = "apply different preprocessing to different columns".

    - Numerical columns  -> StandardScaler   (mean 0, standard deviation 1)
    - Categorical columns -> OneHotEncoder   (e.g. sex=F/M -> sex_F, sex_M columns)

    handle_unknown="ignore" means a category never seen in training will not crash
    the model at prediction time (it simply becomes all zeros).

    IMPORTANT (data leakage): this object is put INSIDE a Pipeline, so it is only
    fitted (it only learns means/categories) on the training data.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """80% training / 20% test split with a fixed random seed."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_clean_split():
    """Convenience function: load -> clean -> X, y -> train/test split."""
    df = clean_data(load_raw_data())
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return df, X_train, X_test, y_train, y_test


def save_processed_data(df, path=PROCESSED_DATA_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    raw = load_raw_data()
    print("=== RAW DATA AUDIT ===")
    report = audit_data(raw)
    print(f"Rows: {report['n_rows']}, Columns: {report['n_columns']}")
    print(f"Missing values: {report['missing_values_total']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print("Columns stored as text (object):",
          [c for c, t in report["dtypes"].items() if t == "object"])
    print("\nFirst five rows:")
    print(raw.head())

    clean = clean_data(raw)
    print("\n=== AFTER CLEANING ===")
    print(f"Shape: {clean.shape}")
    print("G1/G2/G3 dtypes:", clean[["G1", "G2", "G3"]].dtypes.to_dict())
    print("\nDescriptive statistics (numeric):")
    print(clean.describe().T[["mean", "50%", "std", "min", "max"]].round(2))

    print("\nOutliers by IQR rule:")
    for col in ["age", "absences", "failures", "G3"]:
        out = find_outliers_iqr(clean[col])
        print(f"  {col}: {len(out)} outliers, values: {sorted(out.unique().tolist())}")

    print(f"\nStudents with G3 = 0: {(clean['G3'] == 0).sum()}")
    saved = save_processed_data(clean)
    print(f"\nClean data saved to {saved}")
