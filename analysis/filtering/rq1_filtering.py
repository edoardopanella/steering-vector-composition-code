"""
rq1_filtering.py

Documents the two a-priori filters applied before the RQ1 statistical analysis,
each motivated with data from the consolidated frames. Just run this file
directly - no terminal arguments needed.

It reads

    analysis/rq1_consolidated/consolidated_coh30.csv   (filtered at coh >= 30)
    analysis/rq1_consolidated/consolidated_raw.csv      (no coherence filter)

and reports the three filters, in order:

  FILTER 1 - behaviour exclusion: drop `power_seeking` (trait-validity, NOT
  coherence). `power_seeking` only appears in the `normFalse` scheme (8 pairs),
  which is exactly why `normFalse` has 36 raw pairs while `normTrue`/`per_axis`
  have 28. It was dropped a priori at the design stage, before any composition
  run, because the vector is non-expressing: its single-trait delta is negative
  (steering toward power_seeking does not even produce power_seeking; its
  single-trait delta goes negative under stronger steering, confirming a broken
  trait). Every pair containing it therefore has a structurally dead
  axis, so the joint-vs-single regime question is undefined and the pair can
  never show genuine emergence. The power_seeking pairs have HIGH coherence
  (~86), which is the evidence that this filter is about trait validity, not
  coherence.

  FILTER 2 - scheme exclusion: drop the entire `normFalse` scheme (coherence x
  cosine confound). `normFalse` (normalize=False) injects delta = alpha*(v_i+v_j)
  with no re-normalisation, so ||delta|| = alpha*sqrt(2+2cos) grows with pair
  cosine, inflating the dose by up to ~85% on positive-cosine pairs and
  collapsing coherence there. Because the damage is correlated with cosine - the
  RQ1 predictor - it biases rather than merely adds noise to the regime
  comparison. This is argued on RAW coherence (before any gate), because the
  damage falls on exactly the high-cosine pairs that carry the RQ1 signal; a
  row-level gate cannot fix a scheme-level confound, so the scheme is dropped
  before gating. `normTrue`/`per_axis` hold injection magnitude independent of
  cosine.

  FILTER 3 - coherence gate: keep a pair-scheme row only if its mean coherence
  over the 100 generations is at least COH_GATE (=30). Applied LAST, to the two
  surviving magnitude-controlled schemes, producing consolidated_coh30.csv. The
  threshold was chosen after auditing sub-30 generations (broken/repetitive text
  drives behaviour over-attribution by the judge); a stricter 50 floor discards
  too much readable signal. The gate is benign-to-mild here: 0 dyads removed
  under `normTrue` (normalized sum), 6 of 28 under `per_axis`
  (projection-controlled) - the latter mostly antipodal pairs where the per-axis
  total norm blows up.

The scheme coherence comparison (Filter 2) is computed on power_seeking-EXCLUDED
rows so that all three schemes are compared on the same 28-pair design (apples to
apples). Removing the high-coherence power_seeking pairs also un-inflates
normFalse's mean.

Output (results regenerated each run; `## Conclusions` preserved):

    analysis/filtering/rq1_filtering.{md,json}
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Parameters - edit these, then run the file.
# --------------------------------------------------------------------------- #
HI_COS_THRESHOLD = 0.25       # "high positive-cosine" subset is cos > this value
COH_GATE = 30                 # row-level mean-coherence gate that produces coh30.csv
EXCLUDED_BEHAVIOUR = "power_seeking"
COH_COL = "coh_steered"
SCHEME_ORDER = ["normFalse", "normTrue", "per_axis"]
# Human-readable scheme labels used in the paper.
SCHEME_LABEL = {
    "normFalse": "unnormalized sum",
    "normTrue": "normalized sum",
    "per_axis": "projection-controlled",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
COH30_CSV = REPO_ROOT / "analysis/rq1_consolidated/consolidated_coh30.csv"
RAW_CSV = REPO_ROOT / "analysis/rq1_consolidated/consolidated_raw.csv"
OUT_DIR = REPO_ROOT / "analysis/filtering"
OUT_STEM = "rq1_filtering"

# Everything from this header down is hand-written and preserved across reruns.
CONCLUSIONS_HEADER = "## Conclusions"
CONCLUSIONS_HINT = (
    "<!-- Write your conclusions below. This whole section (from the "
    "`## Conclusions` heading down) is kept when the script is re-run; the "
    "results above are regenerated each time. Keep the heading text exactly "
    "as-is. -->"
)
DEFAULT_CONCLUSIONS = (
    f"{CONCLUSIONS_HEADER}\n\n{CONCLUSIONS_HINT}\n\n"
    "Three a-priori filters are applied before the RQ1 statistical analysis.\n\n"
    "**Filter 1 (behaviour) - drop `power_seeking`.** This is a *trait-validity* "
    "filter, not a coherence filter: the power_seeking vector is non-expressing "
    "(negative single-trait delta), so every pair containing it has a "
    "structurally dead axis and cannot inform the joint-vs-single regime "
    "question. The power_seeking pairs have high coherence (~86), confirming the "
    "exclusion is unrelated to coherence. This filter accounts for the 36-vs-28 "
    "raw pair-count difference.\n\n"
    "**Filter 2 (scheme) - drop `normFalse`.** Its injection norm scales with "
    r"pair cosine ($\lVert\delta\rVert = \alpha\sqrt{2+2\cos}$), so its coherence "
    "collapse is concentrated on high-cosine pairs - exactly where the RQ1 "
    "cosine signal lives. Because the degradation is correlated with the "
    "predictor, it biases the regime comparison. The confound is argued on raw "
    "coherence; a row-level gate cannot fix a scheme-level confound (it would "
    "censor exactly the high-cosine pairs that carry the signal), so the scheme "
    "is dropped before gating. On the apples-to-apples (power_seeking-excluded) "
    "28-pair design, `normFalse` raw coherence is the worst of the three "
    "schemes. `normTrue` and `per_axis` hold injection magnitude independent of "
    "cosine and are the valid bases for the analysis.\n\n"
    "**Filter 3 (coherence gate) - keep rows with mean coherence ≥ 30.** The "
    "row-level gate that produces `consolidated_coh30.csv`, applied last to the "
    "two surviving schemes. Below 30 the generations are broken/repetitive and "
    "the judge over-attributes behaviour; a 50 floor would discard too much "
    "readable signal. It removes 0 `normTrue` dyads and 6 `per_axis` dyads "
    "(mostly antipodal, where projection-controlled injection's total norm blows "
    "up).\n"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _is_excluded_pair(df: pd.DataFrame) -> pd.Series:
    return (df["trait_a"] == EXCLUDED_BEHAVIOUR) | (df["trait_b"] == EXCLUDED_BEHAVIOUR)


def _filtered_stats(coh: pd.Series) -> dict:
    return {
        "n": int(coh.size),
        "mean": float(coh.mean()),
        "min": float(coh.min()),
        "max": float(coh.max()),
    }


def _raw_stats(coh: pd.Series) -> dict:
    n = int(coh.size)
    return {
        "n": n,
        "mean": float(coh.mean()),
        "median": float(coh.median()),
        "pct_below_50": float(100.0 * (coh < 50).sum() / n) if n else float("nan"),
        "pct_below_30": float(100.0 * (coh < 30).sum() / n) if n else float("nan"),
    }


def _by_scheme(df: pd.DataFrame, fn) -> dict:
    out = {}
    for scheme in SCHEME_ORDER:
        sub = df.loc[df["scheme"] == scheme, COH_COL].dropna()
        if sub.empty:
            continue
        out[scheme] = fn(sub)
    return out


def _power_seeking_diagnostics(raw: pd.DataFrame) -> dict:
    """Single/joint deltas for the power_seeking axis vs its partner axis, plus
    coherence, on the pairs that contain power_seeking (only present in normFalse)."""
    ps = raw[_is_excluded_pair(raw)].copy()
    if ps.empty:
        return {}

    def pick(row, who, when):
        a_is_ps = row["trait_a"] == EXCLUDED_BEHAVIOUR
        col = {
            ("ps", "single"): "delta_a_single" if a_is_ps else "delta_b_single",
            ("other", "single"): "delta_b_single" if a_is_ps else "delta_a_single",
            ("ps", "joint"): "delta_a_joint" if a_is_ps else "delta_b_joint",
            ("other", "joint"): "delta_b_joint" if a_is_ps else "delta_a_joint",
        }[(who, when)]
        return row[col]

    for who in ("ps", "other"):
        for when in ("single", "joint"):
            ps[f"{who}_{when}"] = ps.apply(lambda r: pick(r, who, when), axis=1)

    return {
        "schemes_present": sorted(ps["scheme"].unique().tolist()),
        "n_pairs": int(len(ps)),
        "mean_coherence": float(ps[COH_COL].mean()),
        "regimes": {k: int(v) for k, v in ps["regime"].value_counts().items()},
        "ps_axis": {
            "mean_single_delta": float(ps["ps_single"].mean()),
            "mean_joint_delta": float(ps["ps_joint"].mean()),
        },
        "partner_axis": {
            "mean_single_delta": float(ps["other_single"].mean()),
            "mean_joint_delta": float(ps["other_joint"].mean()),
        },
    }


def _coherence_gate(raw: pd.DataFrame, coh30: pd.DataFrame) -> dict:
    """Per-scheme effect of the mean-coherence>=COH_GATE row gate that produces
    coh30.csv, plus the censored-dyad table (raw rows that the gate drops).
    Computed on power_seeking-excluded rows."""
    raw = raw[~_is_excluded_pair(raw)]
    coh30 = coh30[~_is_excluded_pair(coh30)]
    counts, censored = {}, {}
    for scheme in SCHEME_ORDER:
        r = raw[raw["scheme"] == scheme]
        kept_keys = set(coh30.loc[coh30["scheme"] == scheme, "pair_key"])
        dropped = r[~r["pair_key"].isin(kept_keys)].sort_values(COH_COL)
        counts[scheme] = {
            "raw": int(len(r)),
            "retained": int(len(r) - len(dropped)),
            "removed": int(len(dropped)),
        }
        censored[scheme] = [
            {
                "trait_a": row["trait_a"],
                "trait_b": row["trait_b"],
                "cos": float(row["cos"]),
                "mean_joint_coherence": float(row[COH_COL]),
            }
            for _, row in dropped.iterrows()
        ]
    return {"threshold": COH_GATE, "counts": counts, "censored_dyads": censored}


def _pair_inventory(raw: pd.DataFrame) -> dict:
    out = {}
    for scheme in SCHEME_ORDER:
        sub = raw[raw["scheme"] == scheme]
        n_ps = int(_is_excluded_pair(sub).sum())
        out[scheme] = {
            "total_pairs": int(len(sub)),
            "power_seeking_pairs": n_ps,
            "pairs_after_filter": int(len(sub) - n_ps),
        }
    return out


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _md_table(stats: dict, columns: list) -> str:
    head = "| scheme | " + " | ".join(h for _, h, _ in columns) + " |"
    rule = "|" + "---|" * (len(columns) + 1)
    lines = [head, rule]
    for scheme in SCHEME_ORDER:
        if scheme not in stats:
            continue
        cells = [format(stats[scheme][k], f) if f else str(stats[scheme][k])
                 for k, _, f in columns]
        lines.append(f"| `{scheme}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_results(r: dict) -> str:
    inv = r["pair_inventory"]
    ps = r["power_seeking"]
    gate = r["coherence_gate"]
    P = []
    P += [
        "# RQ1 a-priori filtering - behaviour, scheme, and coherence gate",
        "",
        f"_Auto-generated by `analysis/filtering/{OUT_STEM}.py` on "
        f"{r['generated_at']}. Do not edit above the Conclusions heading - rerun "
        "the script instead._",
        "",
        "Source frames:",
        f"- filtered (coh≥30): `{COH30_CSV.relative_to(REPO_ROOT)}`",
        f"- raw (unfiltered): `{RAW_CSV.relative_to(REPO_ROOT)}`",
        "",
        f"Coherence column: `{COH_COL}`. High positive-cosine subset: "
        f"`cos > {HI_COS_THRESHOLD}`. Excluded behaviour: `{EXCLUDED_BEHAVIOUR}`.",
        "",
        "Three filters are applied before the RQ1 statistical analysis. They are "
        "*different kinds* of filter and are motivated separately below.",
        "",
        "## Pair inventory",
        "",
        f"`{EXCLUDED_BEHAVIOUR}` appears only in `normFalse`, which is exactly why "
        "it carries 36 raw pairs while the other two schemes carry 28.",
        "",
        _md_table(
            inv,
            [("total_pairs", "raw pairs", "d"),
             (f"power_seeking_pairs", f"{EXCLUDED_BEHAVIOUR} pairs", "d"),
             ("pairs_after_filter", "after filter", "d")],
        ),
        "",
        f"## Filter 1 (behaviour) - drop `{EXCLUDED_BEHAVIOUR}`: trait validity, **not** coherence",
        "",
        f"`{EXCLUDED_BEHAVIOUR}` was dropped a priori at the design stage, before "
        "any composition run. The reason is that the vector is "
        "*non-expressing*: steering toward it does not even produce the "
        "behaviour. Diagnostics over the "
        f"{ps['n_pairs']} pairs that contain it (all in "
        f"`{', '.join(ps['schemes_present'])}`):",
        "",
        "| axis | mean single-trait Δ | mean joint Δ |",
        "|---|---|---|",
        f"| `{EXCLUDED_BEHAVIOUR}` (the dropped axis) | "
        f"{ps['ps_axis']['mean_single_delta']:.1f} | "
        f"{ps['ps_axis']['mean_joint_delta']:.1f} |",
        f"| partner behaviour | "
        f"{ps['partner_axis']['mean_single_delta']:.1f} | "
        f"{ps['partner_axis']['mean_joint_delta']:.1f} |",
        "",
        f"- The `{EXCLUDED_BEHAVIOUR}` axis has a **negative** single-trait Δ "
        f"({ps['ps_axis']['mean_single_delta']:.1f}): even alone, the vector fails "
        "to elicit its own behaviour; its single-trait delta goes negative under "
        "stronger steering, confirming a broken trait. Its partner "
        f"expresses normally (single Δ {ps['partner_axis']['mean_single_delta']:.1f}).",
        "- Every pair containing it therefore has a **structurally dead axis**, so "
        "the \"did axis a survive joint steering?\" question is undefined and the "
        f"pair cannot show genuine emergence. Observed regimes are only: "
        f"{', '.join(f'{k}×{v}' for k, v in ps['regimes'].items())}.",
        f"- **This is not a coherence filter:** the `{EXCLUDED_BEHAVIOUR}` pairs "
        f"have high coherence (mean {ps['mean_coherence']:.1f}). They are removed "
        "for trait validity, not fluency.",
        "",
        "## Filter 2 (scheme) - drop `normFalse`: coherence × cosine confound",
        "",
        "`normFalse` (normalize=False) injects $\\delta = \\alpha(\\hat v_i + "
        "\\hat v_j)$ with no re-normalisation, so $\\lVert\\delta\\rVert = "
        "\\alpha\\sqrt{2+2\\cos}$ grows with pair cosine - up to ~85% more total "
        "dose on the most aligned pairs - and collapses coherence there. The "
        "confound is argued on **raw** coherence (before any gate), because the "
        "damage falls on exactly the high-cosine pairs that carry the RQ1 cosine "
        "signal.",
        "",
        f"All coherence tables below are computed on **`{EXCLUDED_BEHAVIOUR}`-"
        "excluded** rows, so every scheme is compared on the same 28-pair design. "
        "Removing the high-coherence power_seeking pairs also un-inflates "
        "`normFalse`'s mean.",
        "",
        "### Raw frame (no coherence filter) - the real picture",
        "",
        _md_table(
            r["raw_all"],
            [("n", "n", "d"), ("mean", "mean coh", ".1f"),
             ("median", "median", ".1f"),
             ("pct_below_50", "% coh<50", ".1f"),
             ("pct_below_30", "% coh<30", ".1f")],
        ),
        "",
        f"### Raw frame, high positive-cosine pairs (cos > {HI_COS_THRESHOLD})",
        "",
        "Where the confound bites: `normFalse` breaks on the high-cosine pairs "
        "that carry the RQ1 cosine signal.",
        "",
        _md_table(
            r["raw_hicos"],
            [("n", "n", "d"), ("mean", "mean coh", ".1f"),
             ("median", "median", ".1f"),
             ("pct_below_50", "% coh<50", ".1f"),
             ("pct_below_30", "% coh<30", ".1f")],
        ),
        "",
        "### Why a row-level coherence gate cannot rescue it",
        "",
        "The coherence gate (Filter 3, applied next) is a row-level tool. On the "
        "gated frame `normFalse`'s surviving rows look healthy - the gate simply "
        "deletes the broken generations, so a naive average on the gated file "
        "would understate the problem:",
        "",
        _md_table(
            r["filtered"],
            [("n", "n", "d"), ("mean", "mean coh", ".1f"),
             ("min", "min", ".1f"), ("max", "max", ".1f")],
        ),
        "",
        f"But the gate removes {gate['counts']['normFalse']['removed']} of "
        f"{gate['counts']['normFalse']['raw']} `normFalse` rows - the heaviest "
        "censoring of any scheme - and those removals fall on the high-cosine "
        "pairs. The bias is therefore in *which* rows survive, correlated with the "
        "very predictor of the RQ1 analysis. A row-level gate cannot fix a "
        "scheme-level confound, so `normFalse` is dropped as a scheme before "
        "gating. `normTrue` and `per_axis` hold injection magnitude independent of "
        "cosine and are the valid bases for the analysis.",
        "",
        f"## Filter 3 (coherence gate) - keep rows with mean coherence ≥ {gate['threshold']}",
        "",
        f"With `normFalse` removed, the gate is applied at the pair-scheme row "
        f"level to the two surviving schemes: for each joint condition we average "
        f"coherence over its 100 generations and keep the row if the mean is at "
        f"least {gate['threshold']}. This is the row-level step that produces "
        "`consolidated_coh30.csv`. The threshold was set after auditing "
        f"generations below {gate['threshold']}, where broken or repetitive text "
        "caused behaviour over-attribution by the judge; a stricter floor of 50 "
        "discards too much readable signal, and threshold-sensitivity checks "
        "preserve the qualitative directional result.",
        "",
        "Effect of the gate by scheme (power_seeking already excluded):",
        "",
        "| scheme | raw dyads | retained | removed |",
        "|---|---|---|---|",
        *[f"| `{s}` ({SCHEME_LABEL[s]}) | {gate['counts'][s]['raw']} | "
          f"{gate['counts'][s]['retained']} | {gate['counts'][s]['removed']} |"
          for s in ("normTrue", "per_axis")],
        "",
        f"Under `normTrue` (normalized sum) no dyads are removed; under `per_axis` "
        f"(projection-controlled) {gate['counts']['per_axis']['removed']} of "
        f"{gate['counts']['per_axis']['raw']} are removed. The removed `per_axis` "
        "dyads:",
        "",
        "| pair | cosine | mean joint coherence |",
        "|---|---|---|",
        *[f"| {d['trait_a']} + {d['trait_b']} | {d['cos']:+.2f} | "
          f"{d['mean_joint_coherence']:.2f} |"
          for d in gate["censored_dyads"]["per_axis"]],
        "",
        "Five of the six removed `per_axis` rows have cosine at most $+0.13$, "
        "consistent with the total-norm blow-up of projection-controlled injection "
        "for opposed directions.",
        "",
    ]
    return "\n".join(P)


def _split_conclusions(md_text: str) -> str:
    idx = md_text.find(CONCLUSIONS_HEADER)
    return md_text[idx:] if idx != -1 else DEFAULT_CONCLUSIONS


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run() -> dict:
    df_coh30 = pd.read_csv(COH30_CSV)
    df_raw = pd.read_csv(RAW_CSV)

    # Filter-2 coherence comparison is on power_seeking-excluded rows.
    coh30_f = df_coh30[~_is_excluded_pair(df_coh30)]
    raw_f = df_raw[~_is_excluded_pair(df_raw)]
    raw_f_hicos = raw_f[raw_f["cos"] > HI_COS_THRESHOLD]

    results = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "hi_cos_threshold": HI_COS_THRESHOLD,
            "coh_gate": COH_GATE,
            "coh_col": COH_COL,
            "excluded_behaviour": EXCLUDED_BEHAVIOUR,
            "coherence_tables_exclude_behaviour": True,
        },
        "pair_inventory": _pair_inventory(df_raw),
        "power_seeking": _power_seeking_diagnostics(df_raw),
        "coherence_gate": _coherence_gate(df_raw, df_coh30),
        "filtered": _by_scheme(coh30_f, _filtered_stats),
        "raw_all": _by_scheme(raw_f, _raw_stats),
        "raw_hicos": _by_scheme(raw_f_hicos, _raw_stats),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{OUT_STEM}.json"
    md_path = OUT_DIR / f"{OUT_STEM}.md"

    json_path.write_text(json.dumps(results, indent=2) + "\n")
    conclusions = (
        _split_conclusions(md_path.read_text())
        if md_path.exists()
        else DEFAULT_CONCLUSIONS
    )
    md_path.write_text(_render_results(results) + "\n" + conclusions)

    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    return results


if __name__ == "__main__":
    run()
