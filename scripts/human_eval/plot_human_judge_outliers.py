"""Judge-vs-human agreement plots for spotting outliers (layer 17 human eval).

Reads results/human_eval/human_eval_layer17.xlsx (judge_b1/b2, rating_b1/b2 on a
0-100 scale). Each prompt-completion row carries a judge and a human score for
trait a (b1) and trait b (b2); we melt these into one (trait, judge, human) row
per observation, keyed by the ACTUAL trait name (formality / evil / confidence)
rather than the b1/b2 slot, because calibration is trait-dependent.

Pooling all traits onto a single y=x line is misleading (between-trait spread
inflates agreement), so we facet by trait: each panel has its own y=x line and
flags within-trait outliers as |residual| > mean + 2*std of that trait's
residuals. A pooled panel is kept for reference.

Run:
    python -m scripts.human_eval.plot_human_judge_outliers
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

IN_PATH = Path("results/human_eval/human_eval_layer17.xlsx")
OUT_PNG = Path("results/human_eval/human_judge_outliers.png")

SIGMA = 2.0  # within-trait outlier threshold: |residual| > mean + SIGMA*std


def load_long(path=IN_PATH):
    """One row per (text, trait) observation, keyed by actual trait name."""
    df = pd.read_excel(path)
    recs = []
    for i in range(len(df)):
        for tcol, jcol, hcol in (("behavior_1", "judge_b1", "rating_b1"),
                                 ("behavior_2", "judge_b2", "rating_b2")):
            recs.append({
                "row": i,
                "trait": df.at[i, tcol],
                "pair": f"{df.at[i, 'behavior_1']}+{df.at[i, 'behavior_2']}",
                "judge": float(df.at[i, jcol]),
                "human": float(df.at[i, hcol]),
            })
    out = pd.DataFrame(recs)
    out["residual"] = out["human"] - out["judge"]
    return out


def flag_outliers(g):
    thr = g["residual"].abs().mean() + SIGMA * g["residual"].abs().std()
    return g.assign(outlier=g["residual"].abs() > thr), thr


def _panel(ax, g, title):
    g, thr = flag_outliers(g)
    ax.plot([0, 100], [0, 100], ls="--", c="grey", lw=1, zorder=0)
    ax.scatter(g["judge"], g["human"], s=45, alpha=0.7, c="#1f77b4", edgecolors="none")
    out = g[g["outlier"]]
    ax.scatter(out["judge"], out["human"], s=140, facecolors="none",
               edgecolors="black", linewidths=1.6, zorder=5)
    for _, r in out.iterrows():
        ax.annotate(f"#{r['row']}", (r["judge"], r["human"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    pe = pearsonr(g["judge"], g["human"])[0]
    sp = spearmanr(g["judge"], g["human"])[0]
    ax.text(0.04, 0.96, f"n={len(g)}\nr={pe:.2f}  ρ={sp:.2f}\n|Δ|>{thr:.0f}: {int(out.shape[0])}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
    ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); ax.set_aspect("equal")
    ax.set_xlabel("Judge (0-100)"); ax.set_ylabel("Human (0-100)")
    ax.set_title(title)
    return out


def main() -> None:
    d = load_long()
    traits = sorted(d["trait"].unique())

    fig, axes = plt.subplots(1, len(traits) + 1, figsize=(5 * (len(traits) + 1), 5))

    all_out = []
    for ax, t in zip(axes, traits):
        out = _panel(ax, d[d["trait"] == t].copy(), f"{t}")
        all_out.append(out.assign(trait=t))

    # reference pooled panel
    _panel(axes[-1], d.copy(), "ALL traits pooled")

    fig.suptitle("Judge vs. human agreement by trait (layer 17) — circled = within-trait outlier",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {OUT_PNG}\n")

    flagged = pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()
    print(f"Within-trait outliers (|residual| > mean + {SIGMA}*std per trait): {len(flagged)}")
    if len(flagged):
        cols = ["trait", "row", "pair", "judge", "human", "residual"]
        print(flagged.sort_values("residual", key=abs, ascending=False)[cols]
              .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
