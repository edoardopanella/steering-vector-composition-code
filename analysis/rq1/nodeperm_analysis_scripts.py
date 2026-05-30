"""
nodeperm_analysis_scripts.py

Worker module for the RQ1 per-scheme decomposition with an **exact 8!
node-permutation (MRQAP) test**. No argparse: configure and launch it from
`nodeperm_analysis.py`.

This is the script port of `analysis/notebooks/rq1_perscheme_nodeperm.ipynb`.
The model, frame and estimates are identical to the existing per-scheme
decomposition; only the inference is the dyadic MRQAP null.

Why MRQAP (node permutation) rather than edge-wise cosine shuffling
------------------------------------------------------------------
The 28 dyads share behaviour nodes - with 8 behaviours, each appears in 7
pairs. Edge-wise cosine shuffling treats rows as exchangeable, which is exactly
the dependence structure we need to defend against. Node permutation respects
the dyadic dependence: we relabel the 8 behaviours via a permutation π, apply π
to both rows and columns of the cosine matrix simultaneously, keep everything
else fixed at its observed (i, j) positions, and refit. With 8 behaviours,
8! = 40,320 permutations is exhaustive - the resulting null distribution is
*exact*, not Monte-Carlo.

Recipe (per scheme)
-------------------
  1. Build 8×8 symmetric matrices for cos, each outcome Y, mean_single_abs,
     max_single_abs, sem_sim and the coh mask.
  2. Apply the coh mask (per-axis: 6 cells masked out; normTrue: none).
  3. Fit the observed standardized regression y ~ mean_single + max_single + cos
     on masked upper-triangle cells; get β_cos^obs.
  4. For each of the 8! relabelings π, compute cos_perm = cos[π, π], extract the
     same masked cells, refit, get β_cos^π. Y, mean_single, max_single and the
     mask stay fixed at their original positions.
  5. Exact two-sided p = fraction of |β_cos^π| ≥ |β_cos^obs|.

Sections, in order:
  §1  Main result: β_cos + exact node-perm p (both schemes × both outcomes)
  §2  Edge-wise (10k MC) vs node-perm (exact 8!) p side by side
  §3  Robustness on DIRECTION: + sem_sim, |cos| substituted, leave-one-trait-out

Outputs (one combined report covering both schemes):
  - perscheme_nodeperm_analysis.md    human-readable report
  - perscheme_nodeperm_analysis.json  machine-readable results

Public entry point:
    run_nodeperm_analysis()
    run_nodeperm_analysis(n_edge_perm=2000)   # faster edge-wise comparison
"""

import json
import math
import re
import warnings
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything from this header onward is hand-written and preserved across reruns;
# everything above it (the results) is auto-generated and overwritten each run.
CONCLUSIONS_HEADER = "## Conclusions"
CONCLUSIONS_HINT = (
    "<!-- Write your conclusions below. This whole section (from the "
    "`## Conclusions` heading down) is kept when the script is re-run; the results "
    "above are regenerated each time. Keep the heading text exactly as-is. -->"
)
DEFAULT_CONCLUSIONS = f"{CONCLUSIONS_HEADER}\n\n{CONCLUSIONS_HINT}\n\n_(write your conclusions here)_\n"

CONFIG = {
    "csv": REPO_ROOT / "analysis/rq1_consolidated/consolidated_coh30.csv",
    "out_dir": REPO_ROOT / "analysis/results/RQ1/perscheme_nodeperm_analysis",
    "out_stem": "perscheme_nodeperm_analysis",
    "title": "RQ1 per-scheme decomposition - exact 8! node-permutation (MRQAP) test",
    # (CSV scheme value, display label)
    "schemes": [
        ("normTrue", "v2 / normTrue"),
        ("per_axis", "v3 / per_axis"),
    ],
    "outcomes": [
        ("MAGNITUDE (mean_joint_abs)", "mag"),
        ("DIRECTION (supp_mean_signed)", "dir"),
    ],
}


# --------------------------------------------------------------------------- #
# Markdown helpers (no tabulate dependency)
# --------------------------------------------------------------------------- #
def _esc(s):
    return str(s).replace("|", r"\|")


def _fmt(v, floatfmt):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return format(float(v), floatfmt) if floatfmt is not None else f"{float(v):g}"
    return _esc(v)


def df_to_md(df, floatfmt=None, index=False, index_name=None):
    cols = list(df.columns)
    headers = ([index_name or (df.index.name or "")] if index else []) + [str(c) for c in cols]
    headers = [_esc(h) for h in headers]
    rows = []
    for idx, row in df.iterrows():
        cells = ([_esc(idx)] if index else []) + [_fmt(row[c], floatfmt) for c in cols]
        rows.append(cells)
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for cells in rows:
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Data loading + scheme trait set
# --------------------------------------------------------------------------- #
def load_frame(csv_path):
    """Load the consolidated frame and add the signed-suppression outcome."""
    df = pd.read_csv(csv_path)
    df["supp_signed_a"] = df["delta_a_joint"] - df["delta_a_single"]
    df["supp_signed_b"] = df["delta_b_joint"] - df["delta_b_single"]
    df["supp_mean_signed"] = 0.5 * (df["supp_signed_a"] + df["supp_signed_b"])
    return df


def scheme_traits(sub):
    """Per-scheme trait set actually present in the subset.

    IMPORTANT: the trait set is PER SCHEME (the 8-trait set; power_seeking was
    dropped during validation). The pooled union would be 9, but permuting
    power_seeking into v2/v3 cells produces NaN cosines and silently biases the
    p-value toward significance. Each scheme uses the traits it actually has.
    """
    return sorted(set(sub["trait_a"]).union(sub["trait_b"]))


# --------------------------------------------------------------------------- #
# Matrix building + node-permutation engine (verbatim port of the notebook)
# --------------------------------------------------------------------------- #
def build_matrices(sub, traits):
    n = len(traits)
    idx = {t: i for i, t in enumerate(traits)}
    M_cos = np.full((n, n), np.nan)
    M_mag = np.full((n, n), np.nan)
    M_dir = np.full((n, n), np.nan)
    M_ms = np.full((n, n), np.nan)   # mean_single_abs
    M_Ms = np.full((n, n), np.nan)   # max_single_abs
    M_sem = np.full((n, n), np.nan)  # sem_sim (robustness)
    mask = np.zeros((n, n), bool)
    for _, row in sub.iterrows():
        i, j = idx[row.trait_a], idx[row.trait_b]
        for (a, b) in [(i, j), (j, i)]:
            M_cos[a, b] = row["cos"]
            M_mag[a, b] = row["mean_joint_abs"]
            M_dir[a, b] = row["supp_mean_signed"]
            M_ms[a, b] = row["mean_single_abs"]
            M_Ms[a, b] = row["max_single_abs"]
            M_sem[a, b] = row["sem_sim"]
            mask[a, b] = True
    return dict(cos=M_cos, mag=M_mag, dir=M_dir, ms=M_ms, Ms=M_Ms, sem=M_sem, mask=mask)


def _z(a):
    a = np.asarray(a, float)
    s = a.std(ddof=0)
    return (a - a.mean()) / s if s > 0 else np.zeros_like(a)


def fit_beta_cos(cos_v, ms_v, Ms_v, y_v):
    """Standardize predictors + outcome, fit y ~ ms + Ms + cos, return β_cos."""
    X = np.column_stack([np.ones(len(y_v)), _z(ms_v), _z(Ms_v), _z(cos_v)])
    beta, *_ = np.linalg.lstsq(X, _z(y_v), rcond=None)
    return float(beta[3])


def extract(M, iu, mask_vec, perm=None):
    """Upper-triangle values at masked-in cells; M optionally relabeled by perm."""
    if perm is not None:
        M = M[np.ix_(perm, perm)]
    return M[iu][mask_vec]


def node_perm_p(mats, iu, y_key, extra_predictors=None, return_null=False):
    """Exact n! node-permutation (MRQAP) test on β_cos.

    extra_predictors: dict name -> matrix added as fixed (non-permuted) control.
    """
    n = mats["cos"].shape[0]
    mask_vec = mats["mask"][iu]
    ms_obs = extract(mats["ms"], iu, mask_vec)
    Ms_obs = extract(mats["Ms"], iu, mask_vec)
    y_obs = extract(mats[y_key], iu, mask_vec)
    cos_obs = extract(mats["cos"], iu, mask_vec)

    if extra_predictors is None:
        b_obs = fit_beta_cos(cos_obs, ms_obs, Ms_obs, y_obs)
        null = np.empty(math.factorial(n))
        for k, perm in enumerate(permutations(range(n))):
            cos_p = extract(mats["cos"], iu, mask_vec, perm=np.array(perm))
            null[k] = fit_beta_cos(cos_p, ms_obs, Ms_obs, y_obs)
    else:
        extras = [extract(M, iu, mask_vec) for M in extra_predictors.values()]

        def fit_with_extras(cos_v):
            cols = [_z(ms_obs), _z(Ms_obs)] + [_z(e) for e in extras] + [_z(cos_v)]
            X = np.column_stack([np.ones(len(y_obs))] + cols)
            beta, *_ = np.linalg.lstsq(X, _z(y_obs), rcond=None)
            return float(beta[-1])

        b_obs = fit_with_extras(cos_obs)
        null = np.empty(math.factorial(n))
        for k, perm in enumerate(permutations(range(n))):
            cos_p = extract(mats["cos"], iu, mask_vec, perm=np.array(perm))
            null[k] = fit_with_extras(cos_p)

    p_two = float(np.mean(np.abs(null) >= np.abs(b_obs)))
    if return_null:
        return b_obs, p_two, null
    return b_obs, p_two


def node_perm_p_abscos(mats, iu, y_key):
    """Substitute |cos| for signed cos; everything else as in node_perm_p."""
    mats_abs = dict(mats)
    mats_abs["cos"] = np.abs(mats["cos"])
    return node_perm_p(mats_abs, iu, y_key)


def edge_perm_p(sub, y_col, n_perm, rng=None):
    """Edge-wise (row-exchangeable) cosine-shuffle null on β_cos (Monte-Carlo)."""
    rng = rng or np.random.default_rng(0)

    def beta_cos_std(sub_loc):
        X = np.column_stack([np.ones(len(sub_loc)), _z(sub_loc["mean_single_abs"]),
                             _z(sub_loc["max_single_abs"]), _z(sub_loc["cos"])])
        beta, *_ = np.linalg.lstsq(X, _z(sub_loc[y_col].values), rcond=None)
        return float(beta[3])

    obs = beta_cos_std(sub)
    sub_loc = sub.copy()
    cos_arr = sub_loc["cos"].values.copy()
    null = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(cos_arr)
        sub_loc["cos"] = cos_arr
        null[k] = beta_cos_std(sub_loc)
    return obs, float(np.mean(np.abs(null) >= np.abs(obs)))


def loo_node_perm(sub, traits_full):
    """Leave-one-trait-out node-perm on DIRECTION; each fold uses (n-1)! perms."""
    rows = []
    n0 = len(traits_full)
    iu0 = np.triu_indices(n0, k=1)
    mats_full = build_matrices(sub, traits_full)
    b0, p0 = node_perm_p(mats_full, iu0, "dir")
    rows.append({"drop": "(none)", "n_traits": n0,
                 "n_dyads": int(mats_full["mask"][iu0].sum()),
                 "β_cos": b0, "node-perm p": p0})
    for drop in traits_full:
        keep = [t for t in traits_full if t != drop]
        sub_loo = sub[(sub["trait_a"].isin(keep)) & (sub["trait_b"].isin(keep))]
        nk = len(keep)
        mats_loo = build_matrices(sub_loo, keep)
        iu_loo = np.triu_indices(nk, k=1)
        mask_vec = mats_loo["mask"][iu_loo]
        cos_obs = mats_loo["cos"][iu_loo][mask_vec]
        ms_obs = mats_loo["ms"][iu_loo][mask_vec]
        Ms_obs = mats_loo["Ms"][iu_loo][mask_vec]
        y_obs = mats_loo["dir"][iu_loo][mask_vec]
        if len(y_obs) < 5:
            rows.append({"drop": drop, "n_traits": nk, "n_dyads": len(y_obs),
                         "β_cos": np.nan, "node-perm p": np.nan})
            continue
        b_obs = fit_beta_cos(cos_obs, ms_obs, Ms_obs, y_obs)
        null = np.empty(math.factorial(nk))
        for k, perm in enumerate(permutations(range(nk))):
            cos_p = mats_loo["cos"][np.ix_(perm, perm)][iu_loo][mask_vec]
            null[k] = fit_beta_cos(cos_p, ms_obs, Ms_obs, y_obs)
        p_node = float(np.mean(np.abs(null) >= np.abs(b_obs)))
        rows.append({"drop": drop, "n_traits": nk, "n_dyads": int(mask_vec.sum()),
                     "β_cos": b_obs, "node-perm p": p_node})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-scheme driver
# --------------------------------------------------------------------------- #
def run_scheme(sub, label, outcomes, n_edge_perm):
    traits = scheme_traits(sub)
    n = len(traits)
    iu = np.triu_indices(n, k=1)
    mats = build_matrices(sub, traits)
    n_perm = math.factorial(n)
    print(f"[{label}] traits={n}  dyads={int(mats['mask'][iu].sum())}  perms={n_perm:,}")

    # §1 main + §2 edge-vs-node, per outcome
    main = []
    for outcome_name, y_key in outcomes:
        b_obs, p_node, _ = node_perm_p(mats, iu, y_key, return_null=True)
        mask_vec = mats["mask"][iu]
        cos_obs = extract(mats["cos"], iu, mask_vec)
        y_obs = extract(mats[y_key], iu, mask_vec)
        r_biv, p_biv = stats.pearsonr(cos_obs, y_obs)
        y_col = "mean_joint_abs" if y_key == "mag" else "supp_mean_signed"
        b_edge, p_edge = edge_perm_p(sub, y_col, n_edge_perm)
        assert abs(b_edge - b_obs) < 1e-9, "β estimates should be identical"
        main.append({
            "outcome": outcome_name, "n_dyads": int(mask_vec.sum()),
            "beta_cos": b_obs, "node_perm_p": p_node, "n_perms": n_perm,
            "r_biv": float(r_biv), "p_biv": float(p_biv),
            "edge_p": p_edge, "n_edge_perm": n_edge_perm,
        })

    # §3 robustness (DIRECTION)
    b_sem, p_sem = node_perm_p(mats, iu, "dir", extra_predictors={"sem": mats["sem"]})
    b_signed, p_signed = node_perm_p(mats, iu, "dir")
    b_abs, p_abs = node_perm_p_abscos(mats, iu, "dir")
    loo = loo_node_perm(sub, traits)

    return {
        "label": label, "traits": traits, "n_traits": n, "n_perms": n_perm,
        "n_dyads": int(mats["mask"][iu].sum()),
        "main": main,
        "robust": {
            "sem_sim": {"beta_cos": b_sem, "node_perm_p": p_sem},
            "signed_cos": {"beta_cos": b_signed, "node_perm_p": p_signed},
            "abs_cos": {"beta_cos": b_abs, "node_perm_p": p_abs},
            "loo": loo,
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_markdown(cfg, data, n_edge_perm):
    out = []
    a = out.append
    a(f"# {cfg['title']}\n")
    a(f"_Results auto-generated by `nodeperm_analysis_scripts.py` on {datetime.now():%Y-%m-%d %H:%M}. "
      "Everything down to the `## Conclusions` heading is overwritten on each run; "
      "write your own interpretation under that heading and it will be preserved._\n")
    a("Script port of [`rq1_perscheme_nodeperm.ipynb`](../../../notebooks/rq1_perscheme_nodeperm.ipynb). "
      "The model, frame and β estimates are identical to the existing per-scheme decomposition; "
      "only the inference changes - from edge-wise cosine shuffling to the dyadic **MRQAP** null.\n")
    a("**Why MRQAP / node permutation.** The dyads share behaviour nodes (with 8 behaviours each "
      "appears in 7 pairs). Edge-wise shuffling treats rows as exchangeable - exactly the dependence "
      "we must defend against. Node permutation relabels the 8 behaviours via π and applies it to both "
      "rows and columns of the cosine matrix simultaneously, holding Y, the single-behaviour controls "
      "and the coherence mask fixed at their observed (i, j) positions, then refits. With 8 behaviours, "
      "**8! = 40,320** permutations is exhaustive, so the null is *exact*, not Monte-Carlo.\n")

    # Setup
    a("## Setup\n")
    a(f"- Frame: `{cfg['csv'].relative_to(REPO_ROOT)}`, coh ≥ 30 (row-level).\n"
      "- Model: `y ~ mean_single_abs + max_single_abs + cos` (all standardized).\n"
      "- No `mechanical_push` (collinear with cos within a single scheme) and no scheme dummy.\n"
      "- Two outcomes: `mean_joint_abs` (magnitude) and `supp_mean_signed` (direction).\n"
      "- Per-scheme 8-trait set (`power_seeking` dropped during validation); the pooled union "
      "would be 9 but permuting an absent trait into a scheme yields NaN cosines and biases p.\n")
    setup_rows = [(d["label"], d["n_traits"], d["n_dyads"], f"{d['n_perms']:,}") for d in data]
    a(df_to_md(pd.DataFrame(setup_rows, columns=["scheme", "n_traits", "n_dyads", "n_perms (=n!)"])) + "\n")

    # §1 main
    a("## §1 - Main result with node-perm p-values\n")
    a("β estimates are unchanged from the existing per-scheme analysis; the node-perm p is exact. "
      "`r(cos, y)` is the bivariate Pearson correlation for context.\n")
    rows = []
    for d in data:
        for m in d["main"]:
            rows.append({"scheme": d["label"], "outcome": m["outcome"], "n_dyads": m["n_dyads"],
                         "β_cos (std)": m["beta_cos"], "node-perm p (n!)": m["node_perm_p"],
                         "r(cos,y) biv": m["r_biv"], "p biv": m["p_biv"]})
    a(df_to_md(pd.DataFrame(rows), floatfmt=".4f") + "\n")

    # §2 comparison
    a("## §2 - Edge-wise vs node-permutation p-values\n")
    a(f"Same β (same fit); only the null changes. Edge-wise = {n_edge_perm:,} Monte-Carlo cosine "
      "shuffles; node-perm = exact n!.\n")
    rows = []
    for d in data:
        for m in d["main"]:
            rows.append({"scheme": d["label"], "outcome": m["outcome"].split(" ")[0],
                         "β_cos (std)": m["beta_cos"], "edge p (MC)": m["edge_p"],
                         "node-perm p (n!)": m["node_perm_p"],
                         "difference": m["node_perm_p"] - m["edge_p"]})
    a(df_to_md(pd.DataFrame(rows), floatfmt=".4f") + "\n")

    # §3 robustness
    a("## §3 - Robustness on DIRECTION (node-perm p)\n")
    a("Three checks under the dyadic null: (a) add `sem_sim` as a fixed control, "
      "(b) substitute `|cos|` for signed cos, (c) leave-one-trait-out (each fold exhaustive at (n-1)!).\n")
    a("**(a) + sem_sim control** &nbsp; **(b) |cos| vs signed cos**\n")
    rows = []
    for d in data:
        rb = d["robust"]
        rows.append({"scheme": d["label"],
                     "signed cos β": rb["signed_cos"]["beta_cos"],
                     "signed cos p": rb["signed_cos"]["node_perm_p"],
                     "+sem_sim β": rb["sem_sim"]["beta_cos"],
                     "+sem_sim p": rb["sem_sim"]["node_perm_p"],
                     "|cos| β": rb["abs_cos"]["beta_cos"],
                     "|cos| p": rb["abs_cos"]["node_perm_p"]})
    a(df_to_md(pd.DataFrame(rows), floatfmt=".4f") + "\n")
    a("**(c) Leave-one-trait-out** (drop one behaviour; matrix shrinks to (n-1)×(n-1), "
      "permutations to (n-1)!).\n")
    for d in data:
        a(f"**{d['label']}**\n")
        a(df_to_md(d["robust"]["loo"], floatfmt=".4f") + "\n")

    return "\n".join(out)


def _preserve_conclusions(md_path):
    if md_path.exists():
        text = md_path.read_text()
        m = re.search(rf"^{re.escape(CONCLUSIONS_HEADER)}", text, re.MULTILINE)
        if m:
            return text[m.start():].rstrip() + "\n"
    return DEFAULT_CONCLUSIONS


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def build_json(cfg, data, n_edge_perm):
    return {
        "report": "perscheme_nodeperm",
        "source_csv": str(cfg["csv"].relative_to(REPO_ROOT)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_edge_perm": n_edge_perm,
        "schemes": [{
            "label": d["label"], "traits": d["traits"],
            "n_traits": d["n_traits"], "n_dyads": d["n_dyads"], "n_perms": d["n_perms"],
            "main": d["main"],
            "robust": {
                "sem_sim": d["robust"]["sem_sim"],
                "signed_cos": d["robust"]["signed_cos"],
                "abs_cos": d["robust"]["abs_cos"],
                "loo": d["robust"]["loo"].to_dict(orient="records"),
            },
        } for d in data],
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_nodeperm_analysis(n_edge_perm=10000):
    """Run the per-scheme exact node-permutation (MRQAP) analysis and write the report.

    Parameters
    ----------
    n_edge_perm : int
        Monte-Carlo iteration count for the edge-wise comparison in §2 (the
        node-perm test itself is always exhaustive).
    """
    cfg = CONFIG
    print(f"Loading consolidated frame {cfg['csv']}")
    df_all = load_frame(cfg["csv"])

    data = []
    for scheme_value, label in cfg["schemes"]:
        print(f"\n########## scheme: {label} ({scheme_value}) ##########")
        sub = df_all[df_all["scheme"] == scheme_value].reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"no rows for scheme={scheme_value!r}")
        data.append(run_scheme(sub, label, cfg["outcomes"], n_edge_perm))

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{cfg['out_stem']}.md"
    json_path = out_dir / f"{cfg['out_stem']}.json"

    results_md = build_markdown(cfg, data, n_edge_perm)
    md = results_md.rstrip() + "\n\n" + _preserve_conclusions(md_path)
    md_path.write_text(md)
    with open(json_path, "w") as f:
        json.dump(build_json(cfg, data, n_edge_perm), f, indent=2, default=_json_default)

    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")
    return md_path, json_path
