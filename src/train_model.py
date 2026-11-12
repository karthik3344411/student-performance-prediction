"""
train_model.py
--------------
Phases 4-7: model development, evaluation, tuning and saving.

Workflow (in simple words):
1. Load + clean data, split 80% train / 20% test (seed 42).
2. Build 5 pipelines = preprocessing (ColumnTransformer) + a regression model:
     Dummy (baseline), Linear Regression, Ridge, Decision Tree, Random Forest.
3. 5-fold cross-validation on the TRAINING set only -> compare models fairly.
4. Choose the best model using CV RMSE and MAE (not only R2).
5. Tune it with GridSearchCV (5-fold CV, training set only).
6. Evaluate every model ONCE on the untouched test set (MAE, MSE, RMSE, R2).
7. Make plots (comparison, actual vs predicted, residuals, feature importance).
8. Save the full pipeline to models/best_model.joblib and the metrics to
   models/model_metrics.json (the Streamlit app reads these).

Run it:
    python -m src.train_model
"""

import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    load_clean_split,
)

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
COMPARISON_CSV = PROJECT_ROOT / "reports" / "model_comparison.csv"

PRIMARY, SECONDARY, MUTED = "#2a78d6", "#eb6834", "#898781"

CV_FOLDS = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
CV_SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

# Small, explainable hyperparameter grids (keys use the pipeline step name "model__")
PARAM_GRIDS = {
    "Ridge Regression": {"model__alpha": [0.1, 1, 10, 30, 100]},
    "Decision Tree": {
        "model__max_depth": [2, 3, 4, 5, 6, 8],
        "model__min_samples_leaf": [1, 5, 10, 20],
    },
    "Random Forest": {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_features": [1.0, 0.5, "sqrt"],
    },
}


def get_models():
    """The models we compare. Each is later wrapped in the same preprocessing pipeline."""
    return {
        "Dummy Regressor (baseline)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }


def make_pipeline(model):
    """Preprocessing + model in ONE object -> no data leakage, easy to save."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_pipeline(pipeline, X_train, y_train):
    scores = cross_validate(pipeline, X_train, y_train, cv=CV_FOLDS, scoring=CV_SCORING)
    return {
        "CV_MAE": float(-scores["test_mae"].mean()),
        "CV_RMSE": float(-scores["test_rmse"].mean()),
        "CV_RMSE_std": float(scores["test_rmse"].std()),
        "CV_R2": float(scores["test_r2"].mean()),
    }


def get_feature_names(fitted_pipeline):
    return list(fitted_pipeline.named_steps["preprocessor"].get_feature_names_out())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
    })


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)


def plot_model_comparison(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    names = table.index[::-1]
    y = np.arange(len(names))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        h = 0.36
        b1 = ax.barh(y + h / 2, table.loc[names, f"CV_{metric}"], height=h, color=PRIMARY,
                     label="5-fold CV (train)")
        b2 = ax.barh(y - h / 2, table.loc[names, metric], height=h, color=SECONDARY,
                     label="Test set")
        ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=8)
        ax.set_yticks(y, names)
        ax.set(title=f"{metric} (lower is better)", xlabel=f"{metric} (grade points)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 13. Model comparison", fontweight="bold")
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, "13_model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, color=PRIMARY, alpha=0.7, s=40, edgecolor="white", linewidth=1)
    ax.plot([0, 20], [0, 20], color=MUTED, linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set(xlim=(-0.5, 20.5), ylim=(-0.5, 20.5), xlabel="Actual G3", ylabel="Predicted G3",
           title=f"Figure 14. Actual vs predicted G3\n({model_name}, test set)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "14_actual_vs_predicted.png")


def plot_residuals(y_test, y_pred, model_name):
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(y_pred, residuals, color=PRIMARY, alpha=0.7, s=40, edgecolor="white", linewidth=1)
    axes[0].axhline(0, color=MUTED, linestyle="--", linewidth=1.5)
    axes[0].set(xlabel="Predicted G3", ylabel="Residual (actual - predicted)",
                title="Residuals vs predicted")
    axes[1].hist(residuals, bins=20, color=PRIMARY, edgecolor="white", linewidth=1.5)
    axes[1].axvline(0, color=MUTED, linestyle="--", linewidth=1.5)
    axes[1].set(xlabel="Residual (actual - predicted)", ylabel="Number of students",
                title="Residual distribution")
    fig.suptitle(f"Figure 15. Residual analysis ({model_name}, test set)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "15_residuals.png")


def plot_feature_importance(series, title, filename, xlabel):
    top = series.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=PRIMARY, height=0.6)
    ax.set(title=title, xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    _save(fig, filename)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train_and_evaluate(verbose=True):
    _style()
    df, X_train, X_test, y_train, y_test = load_clean_split()
    log = print if verbose else (lambda *a, **k: None)
    log(f"Training rows: {len(X_train)}, test rows: {len(X_test)}, features: {X_train.shape[1]}")

    # ---- Step 1: cross-validate + test every model --------------------------
    rows, fitted = {}, {}
    for name, model in get_models().items():
        pipe = make_pipeline(model)
        cv = cross_validate_pipeline(pipe, X_train, y_train)
        pipe.fit(X_train, y_train)
        test = regression_metrics(y_test, pipe.predict(X_test))
        train = regression_metrics(y_train, pipe.predict(X_train))
        rows[name] = {**cv, **test, "Train_RMSE": train["RMSE"]}
        fitted[name] = pipe
        log(f"{name:28s} CV RMSE={cv['CV_RMSE']:.3f}  CV MAE={cv['CV_MAE']:.3f}  "
            f"Test RMSE={test['RMSE']:.3f}  Test MAE={test['MAE']:.3f}  Test R2={test['R2']:.3f}")

    # ---- Step 2: choose the best model using CV (training data only) --------
    # Rank by CV RMSE and CV MAE together (average rank). Ignore the baseline.
    cv_table = pd.DataFrame(rows).T
    candidates = cv_table.drop(index="Dummy Regressor (baseline)")
    rank = candidates["CV_RMSE"].rank() + candidates["CV_MAE"].rank()
    best_name = rank.idxmin()
    # Linear Regression has no hyperparameters to tune -> tune Ridge (its regularised version)
    tune_name = "Ridge Regression" if best_name == "Linear Regression" else best_name
    log(f"\nBest model by cross-validation: {best_name} -> tuning: {tune_name}")

    # ---- Step 3: hyperparameter tuning with GridSearchCV --------------------
    grid = GridSearchCV(
        make_pipeline(get_models()[tune_name]),
        param_grid=PARAM_GRIDS[tune_name],
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)  # training data only - the test set is never used here
    tuned_pipe = grid.best_estimator_
    tuned_cv = cross_validate_pipeline(tuned_pipe, X_train, y_train)
    tuned_test = regression_metrics(y_test, tuned_pipe.predict(X_test))
    tuned_train = regression_metrics(y_train, tuned_pipe.predict(X_train))
    tuned_label = f"{tune_name} (tuned)"
    rows[tuned_label] = {**tuned_cv, **tuned_test, "Train_RMSE": tuned_train["RMSE"]}
    fitted[tuned_label] = tuned_pipe
    best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
    log(f"Best parameters: {best_params}")
    log(f"Tuned  CV RMSE={tuned_cv['CV_RMSE']:.3f}  Test RMSE={tuned_test['RMSE']:.3f}  "
        f"Test MAE={tuned_test['MAE']:.3f}  Test R2={tuned_test['R2']:.3f}")

    # ---- Step 4: final model = whichever of {default, tuned} has lower CV RMSE
    if tuned_cv["CV_RMSE"] <= rows[tune_name]["CV_RMSE"]:
        final_name = tuned_label
    else:
        final_name = tune_name
    final_pipe = fitted[final_name]
    log(f"Final model: {final_name}")

    table = pd.DataFrame(rows).T
    column_order = ["MAE", "MSE", "RMSE", "R2", "CV_MAE", "CV_RMSE", "CV_RMSE_std", "CV_R2", "Train_RMSE"]
    table = table[column_order]
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(COMPARISON_CSV, index_label="Model")

    # ---- Step 5: plots ------------------------------------------------------
    y_pred = final_pipe.predict(X_test)
    plot_model_comparison(table)
    plot_actual_vs_predicted(y_test, y_pred, final_name)
    plot_residuals(y_test, y_pred, final_name)

    # Permutation importance: how much does test RMSE get worse if we shuffle a feature?
    perm = permutation_importance(final_pipe, X_test, y_test, n_repeats=30,
                                  random_state=RANDOM_STATE,
                                  scoring="neg_root_mean_squared_error", n_jobs=-1)
    perm_series = pd.Series(perm.importances_mean, index=X_test.columns)
    plot_feature_importance(perm_series, "Figure 16. Permutation importance (test set)",
                            "16_permutation_importance.png",
                            "Increase in RMSE when the feature is shuffled")

    model_step = final_pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        imp = pd.Series(model_step.feature_importances_, index=get_feature_names(final_pipe))
        imp.index = [n.split("__", 1)[1] for n in imp.index]
        plot_feature_importance(imp, "Figure 17. Tree-based feature importance (impurity)",
                                "17_tree_feature_importance.png", "Mean decrease in impurity")
        model_importance = imp.sort_values(ascending=False).head(15).to_dict()
    elif hasattr(model_step, "coef_"):
        coefs = pd.Series(model_step.coef_, index=get_feature_names(final_pipe))
        coefs.index = [n.split("__", 1)[1] for n in coefs.index]
        model_importance = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15).to_dict()
    else:
        model_importance = {}

    # Linear regression coefficients (for interpretation in the report)
    lin = fitted["Linear Regression"]
    lin_coefs = pd.Series(lin.named_steps["model"].coef_, index=get_feature_names(lin))
    lin_coefs.index = [n.split("__", 1)[1] for n in lin_coefs.index]
    lin_coefs = lin_coefs.reindex(lin_coefs.abs().sort_values(ascending=False).index)

    # ---- Step 6: save model + metrics --------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, MODEL_PATH)

    # Default values for the app form = typical training student (median / most common)
    defaults = {c: int(X_train[c].median()) for c in NUMERICAL_FEATURES}
    defaults.update({c: str(X_train[c].mode()[0]) for c in CATEGORICAL_FEATURES})

    test_residuals = np.asarray(y_test) - y_pred
    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "UCI Student Performance - student-mat.csv",
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "cv_folds": 5,
        "features": ALL_FEATURES, "excluded_features": ["G1", "G2"],
        "best_by_cv": best_name, "tuned_model": tune_name,
        "final_model": final_name,
        "best_params": best_params,
        "param_grid": {k.replace("model__", ""): [str(v) for v in vals]
                       for k, vals in PARAM_GRIDS[tune_name].items()},
        "grid_search_best_cv_rmse": float(-grid.best_score_),
        "final_test_metrics": rows[final_name],
        "baseline_test_metrics": rows["Dummy Regressor (baseline)"],
        "comparison": {k: {m: float(v) for m, v in r.items()} for k, r in table.iterrows()},
        "permutation_importance": perm_series.sort_values(ascending=False).to_dict(),
        "model_feature_importance": {k: float(v) for k, v in model_importance.items()},
        "linear_regression_top_coefficients": lin_coefs.head(12).to_dict(),
        "test_residuals": {
            "mean": float(test_residuals.mean()),
            "share_abs_error_le_2": float((np.abs(test_residuals) <= 2).mean()),
            "share_abs_error_le_4": float((np.abs(test_residuals) <= 4).mean()),
            "actual_zero_in_test": int((np.asarray(y_test) == 0).sum()),
            "mae_on_actual_zero": float(np.abs(test_residuals[np.asarray(y_test) == 0]).mean())
            if (np.asarray(y_test) == 0).any() else None,
            "mae_on_actual_nonzero": float(np.abs(test_residuals[np.asarray(y_test) != 0]).mean()),
        },
        "form_defaults": defaults,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    log(f"\nSaved model   -> {MODEL_PATH}")
    log(f"Saved metrics -> {METRICS_PATH}")
    log(f"Saved table   -> {COMPARISON_CSV}")
    log("\n" + table.round(3).to_string())
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
