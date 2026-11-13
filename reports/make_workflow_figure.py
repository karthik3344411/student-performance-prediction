"""Draw the project workflow flowchart -> reports/figures/00_workflow.png

Run:  python reports/make_workflow_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "00_workflow.png"
BLUE, LIGHT, ORANGE, INK = "#2a78d6", "#e8f1fc", "#eb6834", "#0b0b0b"

# (x, y, text, highlight)
steps = [
    (1, 3, "1. Raw data\nstudent-mat.csv\n395 x 33", False),
    (3, 3, "2. Audit\ntypes, missing,\nduplicates", False),
    (5, 3, "3. Cleaning\nrange checks,\noutlier review", False),
    (7, 3, "4. EDA +\nstatistical tests\n(12 figures)", False),
    (7, 1.5, "5. Features / target\nX = 30 features\ny = G3 (G1, G2 dropped)", False),
    (5, 1.5, "6. Train/test split\n80 / 20, seed 42\n316 / 79 students", False),
    (3, 1.5, "7. Pipeline\nScaler + OneHot\n+ model", False),
    (1, 1.5, "8. 5-fold CV\n5 models compared\n(train set only)", False),
    (1, 0, "9. GridSearchCV\nRandom Forest\n54 combos x 5 folds", False),
    (3, 0, "10. Test-set\nevaluation\nMAE, RMSE, R²", True),
    (5, 0, "11. Save pipeline\njoblib\nbest_model.joblib", False),
    (7, 0, "12. Streamlit app\nprediction for\nnew students", True),
]

fig, ax = plt.subplots(figsize=(12, 5.6))
ax.set_xlim(0, 8)
ax.set_ylim(-0.7, 3.7)
ax.axis("off")
W, H = 1.55, 0.95
for x, y, text, hi in steps:
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=BLUE if hi else LIGHT, edgecolor=BLUE, linewidth=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.2,
            color="white" if hi else INK, fontweight="bold" if hi else "normal")

for (x1, y1, *_), (x2, y2, *_) in zip(steps, steps[1:]):
    if y1 == y2:
        dx = W / 2 if x2 > x1 else -W / 2
        start, end = (x1 + dx, y1), (x2 - dx, y2)
    else:
        start, end = (x1, y1 - H / 2), (x2, y2 + H / 2)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 color=ORANGE, linewidth=1.8))

ax.text(4, 3.62, "Figure 0. Project workflow", ha="center", fontsize=13, fontweight="bold")
ax.text(4, -0.66, "All preprocessing is fitted inside the Pipeline on training data only "
        "(no data leakage); the test set is used once, at step 10.",
        ha="center", fontsize=9, color="#52514e")
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
