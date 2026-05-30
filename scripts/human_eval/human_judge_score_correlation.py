'''

Calculate the {pearson, spearman, other?} correlation between
human scores and judge scores

'''


# DATA: require scores are ordered so that score_human_i is scoring
#       the same text that score_judge_i is scoring, for all i

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "results/human_eval/human_eval_layer17.xlsx"
OUT_PATH = ROOT / "results/human_eval/human_judge_correlation.md"

# Judge column -> human column, per trait role (b1 = trait a, b2 = trait b).
SCORE_PAIRS = [("judge_b1", "rating_b1"), ("judge_b2", "rating_b2")]


def load_scores(path=IN_PATH):
    """Return (scores_human, scores_judge) aligned 1:1, pooling both traits.

    Each prompt-completion row carries a judge and a human score for trait a
    (b1) and trait b (b2). We stack both trait roles so every (judge, human)
    pair for the same text is one observation.
    """
    df = pd.read_excel(path)
    human, judge = [], []
    for jcol, hcol in SCORE_PAIRS:
        judge.append(pd.to_numeric(df[jcol], errors="raise").to_numpy(float))
        human.append(pd.to_numeric(df[hcol], errors="raise").to_numpy(float))
    return np.concatenate(human), np.concatenate(judge)


def correlate(scores_human, scores_judge):
    h = np.asarray(scores_human, dtype=float)
    j = np.asarray(scores_judge, dtype=float)
    assert h.shape == j.shape, "human/judge scores must align 1:1"

    pearson_r, pearson_p = pearsonr(h, j)
    spearman_r, spearman_p = spearmanr(h, j)

    return {
        "n": len(h),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def _row(label, res):
    return (
        f"| {label} | {res['n']} | {res['pearson_r']:.3f} | {res['pearson_p']:.3g} "
        f"| {res['spearman_r']:.3f} | {res['spearman_p']:.3g} |\n"
    )


def to_markdown(res):
    return (
        "# Human vs. Judge Score Correlation\n\n"
        f"- **N pairs:** {res['n']}\n\n"
        "| Metric | Coefficient | p-value |\n"
        "|---|---|---|\n"
        f"| Pearson r | {res['pearson_r']:.3f} | {res['pearson_p']:.3g} |\n"
        f"| Spearman rho | {res['spearman_r']:.3f} | {res['spearman_p']:.3g} |\n"
    )


def to_markdown_full(pooled, per_role):
    lines = [
        "# Human vs. Judge Score Correlation\n",
        f"Source: `{IN_PATH}`  (layer 17, 0-100 scale)\n",
        "Pooled = trait a (b1) and trait b (b2) scores stacked into one set of "
        "(judge, human) pairs.\n",
        "| Group | N | Pearson r | Pearson p | Spearman rho | Spearman p |",
        "|---|---|---|---|---|---|",
    ]
    md = "\n".join(lines) + "\n"
    md += _row("Pooled (a + b)", pooled)
    for label, res in per_role:
        md += _row(label, res)
    return md


if __name__ == "__main__":
    df = pd.read_excel(IN_PATH)
    scores_human, scores_judge = load_scores()

    pooled = correlate(scores_human, scores_judge)

    per_role = []
    role_labels = {"judge_b1": "Trait a (b1)", "judge_b2": "Trait b (b2)"}
    for jcol, hcol in SCORE_PAIRS:
        res = correlate(
            pd.to_numeric(df[hcol], errors="raise").to_numpy(float),
            pd.to_numeric(df[jcol], errors="raise").to_numpy(float),
        )
        per_role.append((role_labels[jcol], res))

    md = to_markdown_full(pooled, per_role)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(md)
    print(md)
    print(f"Saved to {OUT_PATH}")



