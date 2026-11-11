"""Draw the charts in images/ from the challenge's train.csv, in a light and a dark version each.

The AutoML champion stays in the Azure ML workspace, so the threshold and SHAP charts use the local
stand-in ensemble from validate_locally.py (the same model behind the README's out-of-sample table):

- Feature_Importance: the notebook's random-forest baseline on the no-age features (same code and seed).
- Precision_recall_curve: 10-fold cross-validated scores, so every customer is scored by a model that
  never saw them. The notebook's version scored the training data, which overstates recall (README section 5).
- SHAP: KernelExplainer with a 100-row background on 300 sampled customers, as in the notebook.

Usage (train.csv from the data challenge, not included in this repo; SHAP takes a few minutes):
    python make_charts.py path/to/train.csv

Each chart is written as images/<name>.png (light) and images/<name>-dark.png. The README shows the dark one
under GitHub's dark theme with <picture><source media="(prefers-color-scheme: dark)" ...>.
Colors are the brand chart palettes (validated for contrast and color-vision deficiency).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402

from validate_locally import CUTOFF, best_f2_cutoff, engineer_features, ensemble  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")

THEMES = {
    "light": dict(surface="#f7f9fb", ink="#0e1a2b", ink_2="#3c5a7d", muted="#8a9bb0", grid="#dde4ec",
                  series=["#00948c", "#c2760a", "#5b6fd8"], highlight="#c2760a", accent="#00948c"),
    "dark": dict(surface="#0e1a2b", ink="#f2f5f9", ink_2="#9fb1c7", muted="#6f86a3", grid="#2b3f58",
                 series=["#26a59f", "#c8820f", "#7c80ea"], highlight="#f0a742", accent="#26a59f"),
}
LABELS = {
    "score1": "Score 1", "score2": "Score 2", "monthly_income": "Monthly income",
    "credit_utilization": "Credit utilization", "debt_to_income_ratio": "Debt-to-income ratio",
    "total_open_accounts": "Total open accounts", "number_open_credit_lines": "Open credit lines",
    "number_open_loans": "Open loans", "number_dependants": "Dependants",
    "number_90_days_past_due": "90 days past due", "number_charged_off": "Charged off",
}
# Brand headings are Segoe UI at weight 600, which matplotlib only finds as its own family.
SEMIBOLD = ({"fontfamily": "Segoe UI Semibold", "fontweight": "normal"}
            if any(f.name == "Segoe UI Semibold" for f in font_manager.fontManager.ttflist)
            else {"fontweight": "bold"})


def compute(path):
    """Everything the charts need, from the training data."""
    raw = pd.read_csv(path)
    X, y = engineer_features(raw), raw["target"]

    # Notebook cell 7: random-forest baseline on the no-age features
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1).fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # The same out-of-fold scores as validate_locally.py step 2
    oof = cross_val_predict(ensemble(), X, y, cv=StratifiedKFold(10, shuffle=True, random_state=42),
                            method="predict_proba")[:, 1]
    # The cutoff validate_locally.py step 3 chooses on the 80% development split (the README's 0.09)
    X_dev, _, y_dev, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    dev_oof = cross_val_predict(ensemble(), X_dev, y_dev, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                method="predict_proba")[:, 1]

    # Notebook cell 19, on the stand-in ensemble fitted to all the training data
    model = ensemble().fit(X, y)
    background = shap.sample(X, 100, random_state=42)
    explain = shap.sample(X, 300, random_state=42)
    explainer = shap.KernelExplainer(
        lambda rows: model.predict_proba(pd.DataFrame(rows, columns=X.columns))[:, 1], background)
    return dict(importance=importance, y=y.to_numpy(), oof=oof, chosen=best_f2_cutoff(y_dev, dev_oof),
                shap_values=explainer.shap_values(explain), shap_X=explain)


def threshold_curves(y, proba):
    """Precision, recall, F1 and F2 at each cutoff."""
    cutoffs = np.arange(0.02, 0.5001, 0.0025)
    flagged = proba[None, :] >= cutoffs[:, None]
    tp = (flagged & (y == 1)).sum(axis=1)
    precision = np.divide(tp, flagged.sum(axis=1), out=np.full(len(cutoffs), np.nan), where=flagged.sum(axis=1) > 0)
    recall = tp / (y == 1).sum()
    f1 = 2 * precision * recall / (precision + recall)
    f2 = 5 * precision * recall / (4 * precision + recall)
    return pd.DataFrame(dict(cutoff=cutoffs, precision=precision, recall=recall, f1=f1, f2=f2,
                             flagged=flagged.mean(axis=1)))


def style(t):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink_2"], "text.color": t["ink"],
        "xtick.color": t["ink_2"], "ytick.color": t["ink_2"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titlelocation": "left",
        "axes.titlecolor": t["ink"],
    })


def titles(fig, ax, t, title, subtitle):
    ax.set_title(title, pad=30, **SEMIBOLD)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=t["ink_2"], fontsize=9.5, va="bottom")


def save(fig, name, mode):
    path = os.path.join(OUT, f"{name}.png" if mode == "light" else f"{name}-dark.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"wrote images/{os.path.basename(path)}")


def draw_importance(importance, t, mode):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    order = importance.sort_values()
    ax.barh([LABELS[c] for c in order.index], order.values, height=0.62, color=t["accent"])
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Importance (mean decrease in impurity)")
    top = importance.index[:2]
    titles(fig, ax, t, f"{LABELS[top[0]]} and {LABELS[top[1]]} carry the most weight",
           "Random-forest baseline on the no-age features (notebook cell 7)")
    save(fig, "Feature_Importance", mode)


def draw_threshold(y, oof, chosen, t, mode):
    curves = threshold_curves(y, oof)
    # Stop where under 1% of accounts are flagged; past that, precision is noise from a handful of accounts
    curves = curves[curves["flagged"] >= 0.01]
    plateau = curves.loc[curves["f2"] >= curves["f2"].max() - 0.005, "cutoff"]
    best = curves.iloc[(curves["cutoff"] - chosen).abs().argmin()]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(curves["cutoff"], curves["precision"], color=t["series"][0], linewidth=2, label="Precision")
    ax.plot(curves["cutoff"], curves["recall"], color=t["series"][2], linewidth=2, label="Recall (defaults caught)")
    ax.plot(curves["cutoff"], curves["f1"], color=t["muted"], linewidth=1.5, linestyle=(0, (4, 3)), label="F1")
    ax.plot(curves["cutoff"], curves["f2"], color=t["highlight"], linewidth=2.5, label="F2 (recall weighted 4x)")
    # The brand motif: an amber dot on the optimum
    ax.scatter([best["cutoff"]], [best["f2"]], s=70, color=t["highlight"], edgecolor=t["surface"], linewidth=2,
               zorder=5)
    ax.annotate(f"{chosen:.2f}: the cutoff chosen out of sample\n{best['recall']:.0%} of defaults caught, "
                f"{best['precision']:.0%} precision,\n{best['flagged']:.0%} of accounts flagged",
                (best["cutoff"], best["f2"]), xytext=(0.62, 0.62), textcoords="axes fraction", color=t["ink"],
                fontsize=9.5, va="top",
                arrowprops=dict(arrowstyle="-", color=t["muted"], linewidth=0.8, shrinkA=4, shrinkB=6))
    original = curves.iloc[(curves["cutoff"] - CUTOFF).abs().argmin()]
    ax.axvline(CUTOFF, color=t["muted"], linewidth=1, linestyle=":")
    ax.text(CUTOFF + 0.004, 0.97, f"{CUTOFF:.2f}: original cutoff\n{original['recall']:.0%} caught out of sample",
            color=t["ink_2"], fontsize=9, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, np.ceil(curves["cutoff"].max() * 20) / 20)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Decision cutoff (predicted probability of default)")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.98), frameon=False, labelcolor=t["ink_2"], fontsize=9)
    titles(fig, ax, t, f"Out of sample, F2 is nearly flat from {plateau.min():.2f} to {plateau.max():.2f}",
           f"10-fold cross-validated scores from the local stand-in ensemble, {len(y):,} customers")
    save(fig, "Precision_recall_curve", mode)


def swarm_offsets(values, row_height=0.4, bins=100):
    """SHAP's beeswarm layout: stack points that share a bin, alternating above and below the row."""
    span = values.max() - values.min()
    binned = np.round(bins * (values - values.min()) / (span if span else 1))
    offsets, layer, last = np.zeros(len(values)), 0, None
    for i in np.argsort(binned, kind="stable"):
        layer = layer + 1 if binned[i] == last else 0
        offsets[i] = np.ceil(layer / 2) * (1 if layer % 2 else -1)
        last = binned[i]
    return offsets * 0.9 * row_height / max(np.abs(offsets).max(), 1)


def draw_shap(shap_values, X, t, mode):
    cmap = LinearSegmentedColormap.from_list("feature_value", [t["series"][0], t["muted"], t["series"][1]])
    order = np.argsort(np.abs(shap_values).mean(axis=0))  # least important first, so the top row is the largest
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for row, j in enumerate(order):
        values, feature = shap_values[:, j], X.iloc[:, j].to_numpy(dtype=float)
        low, high = np.percentile(feature, [5, 95])
        if low == high:
            low, high = feature.min(), feature.max()
        shade = np.clip((feature - low) / (high - low), 0, 1) if high > low else np.full(len(feature), 0.5)
        ax.axhline(row, color=t["grid"], linewidth=0.8, zorder=0)
        ax.scatter(values, row + swarm_offsets(values), c=cmap(shade), s=13, linewidth=0.3,
                   edgecolor=t["surface"], zorder=2)
    ax.axvline(0, color=t["muted"], linewidth=1, zorder=1)
    ax.set_yticks(range(len(order)), [LABELS[X.columns[j]] for j in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("SHAP value (change in predicted probability of default)")
    bar = fig.colorbar(ScalarMappable(cmap=cmap), ax=ax, ticks=[0, 1], aspect=40, pad=0.02, fraction=0.03)
    bar.ax.set_yticklabels(["Low", "High"])
    bar.ax.tick_params(length=0, colors=t["ink_2"])
    bar.outline.set_visible(False)
    bar.set_label("Feature value", color=t["ink_2"])
    top = X.columns[order[-1]]
    titles(fig, ax, t, f"{LABELS[top]} moves the predicted default risk the most",
           f"SHAP values for {len(X)} sampled customers, local stand-in ensemble (KernelExplainer)")
    save(fig, "SHAP", mode)


def draw_all(data):
    for mode, t in THEMES.items():
        style(t)
        draw_importance(data["importance"], t, mode)
        draw_threshold(data["y"], data["oof"], data["chosen"], t, mode)
        draw_shap(data["shap_values"], data["shap_X"], t, mode)


if __name__ == "__main__":
    draw_all(compute(sys.argv[1] if len(sys.argv) > 1 else "train.csv"))
