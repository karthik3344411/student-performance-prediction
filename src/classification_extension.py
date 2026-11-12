"""
classification_extension.py
---------------------------
OPTIONAL extension (Syllabus Unit IV - Classification).

A separate experiment: predict PASS (G3 >= 10) vs FAIL (G3 < 10).
Same 30 input features as the regression model (G1/G2 still excluded) and the
same leakage-safe preprocessing pipeline.

Note: the label is made from the same final grade G3, so this is a different
view of the same problem - not an independent early-warning system.

Run:
    python -m src.classification_extension
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    FIGURES_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    clean_data,
    get_features_and_target,
    load_raw_data,
)

PASS_MARK = 10
RESULTS_PATH = PROJECT_ROOT / "reports" / "classification_results.json"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def get_classifiers():
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }


def run_classification(verbose=True):
    df = clean_data(load_raw_data())
    X, g3 = get_features_and_target(df)
    y = (g3 >= PASS_MARK).astype(int)  # 1 = pass, 0 = fail
    # stratify keeps the same pass/fail ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = {}, {}
    for name, clf in get_classifiers().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", clf)])
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows[name] = {
            "CV_F1_macro": float(cv_f1),
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision_fail": float(precision_score(y_test, pred, pos_label=0, zero_division=0)),
            "Recall_fail": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
            "F1_fail": float(f1_score(y_test, pred, pos_label=0, zero_division=0)),
            "F1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_test, proba)),
        }
        fitted[name] = pipe

    table = pd.DataFrame(rows).T
    best = table.drop(index="Dummy (most frequent)")["CV_F1_macro"].idxmax()

    # Figure 18: confusion matrix of the best classifier
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = confusion_matrix(y_test, fitted[best].predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["Fail (<10)", "Pass (>=10)"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.grid(False)
    ax.set_title(f"Figure 18. Confusion matrix\n({best}, test set)", fontweight="bold")
    fig.savefig(FIGURES_DIR / "18_confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Figure 19: ROC curves
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for color, name in zip(COLORS, [n for n in fitted if not n.startswith("Dummy")] + []):
        RocCurveDisplay.from_estimator(fitted[name], X_test, y_test, ax=ax, name=name,
                                       color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#898781", label="Random guess")
    ax.set_title("Figure 19. ROC curves - pass/fail (test set)", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIGURES_DIR / "19_roc_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    results = {
        "pass_mark": PASS_MARK,
        "pass_rate_overall": float(y.mean()),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "best_model": best,
        "confusion_matrix_best": cm.tolist(),
        "comparison": table.to_dict(orient="index"),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Pass rate: {y.mean():.1%}  (train {len(X_train)}, test {len(X_test)})")
        print(table.round(3).to_string())
        print(f"\nBest by CV macro-F1: {best}")
        print("Confusion matrix [[TN(fail->fail), FP], [FN, TP]]:\n", cm)
    return results


if __name__ == "__main__":
    run_classification()
