"""
exploratory_analysis.py
-----------------------
Phase 3 - Exploratory Data Analysis (EDA) and statistical tests.

What this file does:
- Creates every EDA chart and saves it as a PNG in reports/figures/.
- Computes descriptive statistics (mean, median, std, min, max).
- Runs simple statistical tests (t-test, ANOVA, Kruskal-Wallis, Spearman
  correlation, chi-square, 95% confidence interval).
- Saves all numbers to reports/eda_results.json so the report uses real values.

Run it:
    python -m src.exploratory_analysis

Reminder: a correlation or a group difference does NOT prove causation.
"""

import json

import matplotlib

matplotlib.use("Agg")  # draw charts to files (no window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data_preprocessing import (
    FIGURES_DIR,
    NUMERICAL_FEATURES,
    PROJECT_ROOT,
    TARGET,
    clean_data,
    find_outliers_iqr,
    load_raw_data,
)

EDA_RESULTS_PATH = PROJECT_ROOT / "reports" / "eda_results.json"

# One consistent, calm colour style for every chart
PRIMARY = "#2a78d6"    # blue  - main colour
SECONDARY = "#eb6834"  # orange - second group
MUTED = "#898781"
PALETTE_2 = [PRIMARY, SECONDARY]
FLIER = {"markerfacecolor": MUTED, "markeredgecolor": "white", "marker": "o"}  # outlier dots

STUDYTIME_LABELS = {1: "<2 h", 2: "2-5 h", 3: "5-10 h", 4: ">10 h"}


def set_plot_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.edgecolor": "#c3c2b7",
        "grid.color": "#e1e0d9",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def descriptive_statistics(df):
    """Mean, median, std, min, max for every numeric column."""
    numeric = df.select_dtypes(include="number")
    table = pd.DataFrame({
        "mean": numeric.mean(),
        "median": numeric.median(),
        "std": numeric.std(),
        "min": numeric.min(),
        "max": numeric.max(),
    })
    return table.round(2)


# ---------------------------------------------------------------------------
# Charts - univariate (one variable at a time)
# ---------------------------------------------------------------------------
def plot_g3_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df[TARGET], bins=range(0, 22), color=PRIMARY, edgecolor="white", linewidth=2)
    ax.axvline(df[TARGET].mean(), color=SECONDARY, linewidth=2,
               label=f"Mean = {df[TARGET].mean():.2f}")
    ax.axvline(df[TARGET].median(), color="#0b0b0b", linewidth=2, linestyle="--",
               label=f"Median = {df[TARGET].median():.0f}")
    ax.set(title="Figure 1. Distribution of final grade (G3)",
           xlabel="Final grade G3 (0-20)", ylabel="Number of students")
    ax.legend(frameon=False)
    return _save(fig, "01_g3_distribution.png")


def plot_gender_distribution(df):
    counts = df["sex"].value_counts().reindex(["F", "M"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Female (F)", "Male (M)"], counts.values, color=PALETTE_2, width=0.55)
    ax.bar_label(bars, labels=[f"{v} ({v / len(df):.0%})" for v in counts.values], padding=3)
    ax.set(title="Figure 2. Student gender distribution", ylabel="Number of students")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "02_gender_distribution.png")


def plot_studytime_distribution(df):
    counts = df["studytime"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar([STUDYTIME_LABELS[i] for i in counts.index], counts.values,
                  color=PRIMARY, width=0.6)
    ax.bar_label(bars, padding=3)
    ax.set(title="Figure 3. Weekly study time distribution",
           xlabel="Weekly study time", ylabel="Number of students")
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, "03_studytime_distribution.png")


def plot_absences_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].hist(df["absences"], bins=30, color=PRIMARY, edgecolor="white", linewidth=1.5)
    axes[0].set(title="Figure 4a. Absences distribution",
                xlabel="Number of absences", ylabel="Number of students")
    sns.boxplot(y=df["absences"], ax=axes[1], color=PRIMARY, width=0.4,
                flierprops=FLIER)
    axes[1].set(title="Figure 4b. Absences boxplot (outliers)", ylabel="Number of absences")
    return _save(fig, "04_absences_distribution.png")


def plot_outlier_boxplots(df):
    cols = ["age", "failures", "absences", "G3"]
    fig, axes = plt.subplots(1, len(cols), figsize=(12, 3.8))
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color=PRIMARY, width=0.45,
                    flierprops=FLIER)
        ax.set(title=col, ylabel="")
    fig.suptitle("Figure 5. Outlier inspection (IQR boxplots)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "05_outlier_boxplots.png")


# ---------------------------------------------------------------------------
# Charts - bivariate (target vs one variable)
# ---------------------------------------------------------------------------
def plot_g3_by_studytime(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="studytime", y=TARGET, ax=ax, color=PRIMARY, width=0.5,
                flierprops=FLIER)
    means = df.groupby("studytime")[TARGET].mean()
    ax.scatter(range(len(means)), means.values, color=SECONDARY, zorder=5, s=60,
               edgecolor="white", linewidth=2, label="Group mean")
    ax.set_xticks(range(4), [STUDYTIME_LABELS[i] for i in sorted(df["studytime"].unique())])
    ax.set(title="Figure 6. Final grade by weekly study time",
           xlabel="Weekly study time", ylabel="Final grade G3")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, "06_g3_by_studytime.png")


def plot_g3_by_gender(df):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.boxplot(data=df, x="sex", y=TARGET, order=["F", "M"], hue="sex", hue_order=["F", "M"],
                palette=PALETTE_2, ax=ax, width=0.5, legend=False)
    ax.set_xticks([0, 1], ["Female (F)", "Male (M)"])
    ax.set(title="Figure 7. Final grade by gender", xlabel="", ylabel="Final grade G3")
    return _save(fig, "07_g3_by_gender.png")


def plot_absences_vs_g3(df):
    nonzero = df[df[TARGET] > 0]
    zero = df[df[TARGET] == 0]
    rho_all = stats.spearmanr(df["absences"], df[TARGET])[0]
    rho_nz = stats.spearmanr(nonzero["absences"], nonzero[TARGET])[0]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(nonzero["absences"], nonzero[TARGET], color=PRIMARY, alpha=0.55, s=36,
               edgecolor="white", linewidth=0.8, label=f"G3 > 0 (n={len(nonzero)})")
    ax.scatter(zero["absences"], zero[TARGET], color=SECONDARY, s=60, marker="X",
               edgecolor="white", linewidth=0.8,
               label=f"G3 = 0 (n={len(zero)}, all with 0 absences)")
    # simple straight trend line for students with G3 > 0
    slope, intercept = np.polyfit(nonzero["absences"], nonzero[TARGET], 1)
    xs = np.array([0, df["absences"].max()])
    ax.plot(xs, intercept + slope * xs, color="#0b0b0b", linewidth=1.5, linestyle="--",
            label="Linear trend (G3 > 0)")
    ax.set(title="Figure 8. Absences vs final grade",
           xlabel="Number of absences", ylabel="Final grade G3")
    ax.text(0.98, 0.04, f"Spearman rho: all = {rho_all:.2f}, G3 > 0 only = {rho_nz:.2f}",
            transform=ax.transAxes, ha="right", fontsize=9, color="#52514e")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    return _save(fig, "08_absences_vs_g3.png")


def plot_failures_vs_g3(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="failures", y=TARGET, ax=ax, color=PRIMARY, width=0.5,
                flierprops=FLIER)
    counts = df["failures"].value_counts().sort_index()
    ax.set_xticks(range(len(counts)), [f"{i}\n(n={n})" for i, n in counts.items()])
    ax.set(title="Figure 9. Past class failures vs final grade",
           xlabel="Number of past failures", ylabel="Final grade G3")
    return _save(fig, "09_failures_vs_g3.png")


# ---------------------------------------------------------------------------
# Charts - multivariate
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(df):
    cols = NUMERICAL_FEATURES + ["G1", "G2", "G3"]
    corr = df[cols].corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=1,
                linecolor="white", cbar_kws={"label": "Pearson r", "shrink": 0.7}, ax=ax)
    ax.grid(False)
    ax.set_title("Figure 10. Correlation heatmap of numerical variables\n"
                 "(G1/G2 shown for reference only - they are NOT model inputs)")
    return _save(fig, "10_correlation_heatmap.png")


def plot_multivariate_studytime_sex(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.pointplot(data=df, x="studytime", y=TARGET, hue="sex", hue_order=["F", "M"],
                  palette=PALETTE_2, dodge=0.25, errorbar=("ci", 95), markers="o",
                  capsize=0.08, ax=ax)
    ax.set_xticks(range(4), [STUDYTIME_LABELS[i] for i in range(1, 5)])
    ax.set(title="Figure 11. Mean final grade by study time and gender (95% CI)",
           xlabel="Weekly study time", ylabel="Mean final grade G3")
    ax.legend(title="Gender", frameon=False)
    return _save(fig, "11_g3_studytime_gender.png")


def plot_multivariate_higher_failures(df):
    pivot = df.pivot_table(index="higher", columns="failures", values=TARGET, aggfunc="mean")
    counts = df.pivot_table(index="higher", columns="failures", values=TARGET, aggfunc="count")
    annot = pivot.round(1).astype(str) + "\n(n=" + counts.fillna(0).astype(int).astype(str) + ")"
    annot = annot.where(counts.notna(), "")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="Blues", linewidths=2, linecolor="white",
                cbar_kws={"label": "Mean G3"}, ax=ax)
    ax.grid(False)
    ax.set(title="Figure 12. Mean G3 by higher-education wish and past failures",
           xlabel="Past failures", ylabel="Wants higher education")
    return _save(fig, "12_g3_higher_failures.png")


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def mean_confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sem = stats.sem(values)
    low, high = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)
    return float(mean), float(low), float(high)


def run_statistical_tests(df):
    results = {}

    # 1. 95% confidence interval for the average final grade
    mean, low, high = mean_confidence_interval(df[TARGET])
    results["g3_mean_ci95"] = {"mean": mean, "low": low, "high": high}

    # 2. Normality check of G3 (explains why we also use non-parametric tests)
    shapiro = stats.shapiro(df[TARGET])
    results["g3_shapiro"] = {"W": float(shapiro.statistic), "p_value": float(shapiro.pvalue)}

    # 3. Welch t-test: do female and male students have different mean G3?
    female = df.loc[df["sex"] == "F", TARGET]
    male = df.loc[df["sex"] == "M", TARGET]
    t = stats.ttest_ind(female, male, equal_var=False)
    results["ttest_g3_by_sex"] = {
        "mean_F": float(female.mean()), "mean_M": float(male.mean()),
        "n_F": int(len(female)), "n_M": int(len(male)),
        "t_stat": float(t.statistic), "p_value": float(t.pvalue),
    }

    # 4. One-way ANOVA + Kruskal-Wallis: does G3 differ between study-time groups?
    groups = [g[TARGET].values for _, g in df.groupby("studytime")]
    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    results["anova_g3_by_studytime"] = {
        "group_means": {int(k): float(v) for k, v in df.groupby("studytime")[TARGET].mean().items()},
        "group_sizes": {int(k): int(v) for k, v in df.groupby("studytime")[TARGET].size().items()},
        "F_stat": float(anova.statistic), "anova_p_value": float(anova.pvalue),
        "kruskal_H": float(kruskal.statistic), "kruskal_p_value": float(kruskal.pvalue),
    }

    # 5. Kruskal-Wallis for failures groups
    groups = [g[TARGET].values for _, g in df.groupby("failures")]
    kw = stats.kruskal(*groups)
    results["kruskal_g3_by_failures"] = {
        "group_means": {int(k): float(v) for k, v in df.groupby("failures")[TARGET].mean().items()},
        "group_sizes": {int(k): int(v) for k, v in df.groupby("failures")[TARGET].size().items()},
        "H_stat": float(kw.statistic), "p_value": float(kw.pvalue),
    }

    # 6. Spearman correlation of every numeric feature with G3
    spearman = {}
    for col in NUMERICAL_FEATURES:
        rho, p = stats.spearmanr(df[col], df[TARGET])
        pearson_r = df[col].corr(df[TARGET])
        spearman[col] = {"spearman_rho": float(rho), "p_value": float(p),
                         "pearson_r": float(pearson_r)}
    results["correlations_with_g3"] = spearman

    # 6b. The G3 = 0 group: students with a final grade of 0 (likely did not sit the exam)
    zero = df[df[TARGET] == 0]
    nonzero = df[df[TARGET] > 0]
    rho, p = stats.spearmanr(nonzero["absences"], nonzero[TARGET])
    results["g3_zero_analysis"] = {
        "n_zero": int(len(zero)),
        "share_zero_with_zero_absences": float((zero["absences"] == 0).mean()),
        "share_nonzero_with_zero_absences": float((nonzero["absences"] == 0).mean()),
        "mean_failures_zero": float(zero["failures"].mean()),
        "mean_failures_nonzero": float(nonzero["failures"].mean()),
        "n_zero_with_g2_above_0": int((zero["G2"] > 0).sum()),
        "absences_spearman_rho_g3_gt0": float(rho),
        "absences_spearman_p_g3_gt0": float(p),
    }

    # 7. Chi-square: is wanting higher education associated with passing (G3 >= 10)?
    passed = np.where(df[TARGET] >= 10, "pass", "fail")
    table = pd.crosstab(df["higher"], passed)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    results["chi2_higher_vs_pass"] = {
        "table": {str(k): {str(c): int(v) for c, v in row.items()} for k, row in table.iterrows()},
        "chi2": float(chi2), "dof": int(dof), "p_value": float(p),
        "pass_rate_higher_yes": float((df.loc[df["higher"] == "yes", TARGET] >= 10).mean()),
        "pass_rate_higher_no": float((df.loc[df["higher"] == "no", TARGET] >= 10).mean()),
    }

    # 8. Group comparison table (mean G3 by several categories)
    group_means = {}
    for col in ["sex", "address", "internet", "higher", "schoolsup", "paid", "romantic", "school"]:
        g = df.groupby(col)[TARGET].agg(["mean", "count"])
        group_means[col] = {str(k): {"mean": float(r["mean"]), "n": int(r["count"])}
                            for k, r in g.iterrows()}
    results["group_means_g3"] = group_means

    return results


def run_full_eda(df=None, verbose=True):
    """Create every figure and compute every statistic. Returns the results dict."""
    set_plot_style()
    if df is None:
        df = clean_data(load_raw_data())

    figures = [
        plot_g3_distribution(df), plot_gender_distribution(df),
        plot_studytime_distribution(df), plot_absences_distribution(df),
        plot_outlier_boxplots(df), plot_g3_by_studytime(df), plot_g3_by_gender(df),
        plot_absences_vs_g3(df), plot_failures_vs_g3(df), plot_correlation_heatmap(df),
        plot_multivariate_studytime_sex(df), plot_multivariate_higher_failures(df),
    ]

    results = run_statistical_tests(df)
    results["descriptive_statistics"] = descriptive_statistics(df).to_dict(orient="index")
    results["outliers_iqr"] = {
        col: {"count": int(len(find_outliers_iqr(df[col]))),
              "values": sorted(int(v) for v in find_outliers_iqr(df[col]).unique())}
        for col in ["age", "failures", "absences", "G3"]
    }
    results["n_students_g3_zero"] = int((df[TARGET] == 0).sum())
    results["n_students"] = int(len(df))
    results["figures"] = [p.name for p in figures]

    EDA_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EDA_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Saved {len(figures)} figures to {FIGURES_DIR}")
        print(f"Saved statistics to {EDA_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    res = run_full_eda()
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("descriptive_statistics", "figures")}, indent=2))
