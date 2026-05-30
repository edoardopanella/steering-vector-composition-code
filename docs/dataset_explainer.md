# Dataset explainer — composition datasets and what they're for

This document maps every dataset Riccardo generated from **Phase 13 onward** (experiments log, `r-scibo` entries), explains each one, and identifies the **three analysis-ready datasets** you can compute statistics on (compositionality prediction + trajectory analysis). It closes with a proposed cleanup of `results/`.

> **One-line orientation.** There are three full composition runs — **v1 (Phase 12)**, **v2 (Phase 12.5)**, **v3 (Phase 15.16)** — that differ only in *how the two steering vectors are combined into the joint injection*. Each run produces a matched pair of files: a **scoring** dataset (judge scores → compositionality) and a **trajectory** dataset (per-layer projections → mechanism). Those three runs are the datasets you analyse. Everything else Riccardo made (Phase 13, Phase 14, the Phase 15 pilots) is *supporting/diagnostic* work that justified the move from v1 → v2 → v3.

---

## 1. The three analysis-ready datasets

All three share the same skeleton: **Llama-3.1-8B-Instruct, layer 17, unit-normalised vectors, `gpt-4.1-mini` judge.** They differ only in the **joint-injection normalisation mode** and (for v2/v3) the trait set and α. For unit vectors `v̂_i, v̂_j` with `cos = cos(v̂_i, v̂_j)`, the joint perturbation `δ` at setting (1,1) is:

| Run | Phase | Mode | Joint `δ` formula | `‖δ‖` | α | Pairs | Closed-form `π_i^(1,1)−π_i^(1,0)` |
|---|---|---|---|---|---:|---:|---|
| **v1** | 12 | `normalize=False` | `α·(v̂_i + v̂_j)` | `α·√(2+2cos)` (4.0→7.4) | 4.0 | 36 | `α·cos` |
| **v2** | 12.5 | `normalize=True` | `α·(v̂_i + v̂_j)/‖v̂_i + v̂_j‖` | `α` (constant) | 4.5 | 28 | `α·[√((1+cos)/2) − 1]` |
| **v3** | 15.16 | `per_axis` | `(α/(1+cos))·(v̂_i + v̂_j)` | `α·√(2/(1+cos))` | 4.5 | 28 | `0` (exactly) |

Why three modes exist (full reasoning in log Phases 14–15):
- **v1 / `normalize=False`** keeps the *coefficient* on each unit vector constant. Side effect: `‖δ‖` swings from 4 (antipodal) to 7.4 (high-cos), so high-cos joints over-steer and **coherence collapses** (Phase 14 Problem 2: 10/36 pairs had mean joint coh < 30). RQ2 math is clean (`Δπ = α·cos`) but RQ1 ratios are coherence-confounded.
- **v2 / `normalize=True`** holds *total `‖δ‖` = α* constant. Coherence is safe everywhere (no pair below coh 40), at the cost of *under-dosing* high-cos pairs. **This is the committed primary dataset** (Phase 12.5 decision, E15.13).
- **v3 / `per_axis`** holds each behaviour's *per-axis push = α* constant ("fairness": each trait sees the same push it'd see if steered alone). Closed-form joint−single projection is exactly 0, which makes it the **mechanical-null** dataset — the trajectory baseline against which "real" composition effects are measured. Downside: `‖δ‖` blows up on antipodal pairs (up to 8.2), so the most antipodal pair decoheres.

### 1a. Scoring datasets — for *predicting compositionality* (RQ1)

| Run | Summary JSON | Per-CSV dir |
|---|---|---|
| v1 | `results/composition/v1_phase12_normFalse_a4/scoring/summary.json` | `…/v1_phase12_normFalse_a4/scoring/Llama-3.1-8B-Instruct/` (144 CSV) |
| v2 | `results/composition/v2_phase125_normTrue_a4.5/scoring/summary.json` | `…/v2_phase125_normTrue_a4.5/scoring/Llama-3.1-8B-Instruct/` (112 CSV) |
| v3 | `results/composition/v3_phase1516_perAxis_a4.5/scoring/summary.json` | `…/v3_phase1516_perAxis_a4.5/scoring/Llama-3.1-8B-Instruct/` (112 CSV) |

**What's inside.** Each summary JSON has run-level metadata (`alpha`, `composition_mode`, `traits`, `tau_value`, …) plus a `pairs` list — one record per trait pair. Each pair record carries, for the four conditions `baseline / single_a / single_b / steered (joint)`, the judge means: `trait_a_mean`, `trait_b_mean`, `composition_mean`, `coherence_mean`. Then a `delta` block (`trait_a_joint`, `trait_b_joint`, `trait_a_single`, `trait_b_single`, `composition`, `coherence`), a `cos`, a `regime` label (`additive / dominant / suppressive / emergent / mixed`), and an `l17_sanity` block comparing the observed joint−single projection shift against each mode's closed-form prediction.

**What you compute from these.** The RQ1 statistics: regime distribution; **Q-vs-|cos| Spearman** (does geometric overlap predict composition outcome?); coherence-by-regime; per-pair Δ_joint vs Δ_single ratios. The per-condition CSVs (one row per generation, with raw judge scores + coherence) let you re-aggregate with filters (e.g. coh≥30) or recompute per-pair distributions (bimodal vs continuous axes, Phase 14 Problem 4).

> **v3 metadata caveat:** in `..._v3_summary.json` the `composition_mode` field correctly says `per_axis`, but the `injection` string was copied stale from v2 (shows the `normalize=True` formula). The run *is* per_axis — confirmed by `l17_sanity.pred_diff_per_axis = 0.0` and the Phase 15.16 commit. Trust `composition_mode`, not the `injection` string. Worth fixing the string.

### 1b. Trajectory datasets — for *analysing trajectories* (RQ2)

All three live under `…/<version>/trajectories/`: `aggregate.parquet` + `Llama-3.1-8B-Instruct/` (per-pair shards) + `tau.json`.

| Run | trajectories dir | per-pair shards | τ |
|---|---|---|---|
| v1 | `results/composition/v1_phase12_normFalse_a4/trajectories/` | 36 parquet | 0.890 |
| v2 | `results/composition/v2_phase125_normTrue_a4.5/trajectories/` | 28 parquet | 0.830 |
| v3 | `results/composition/v3_phase1516_perAxis_a4.5/trajectories/` | 28 parquet | 0.826 |

(v1's `trajectories/` also holds `composition_delta_ldiv.csv`, the derived L_div table.)

**What's inside.** The aggregate parquet (v3: 268,800 rows) is long-format, one row per (pair, condition, prompt, completion, layer). Columns: `pair_id, behaviour_index, alpha_i, alpha_j, prompt_id, question_id, completion_id, layer, projection_value, trait_i, trait_j, cosine, stratum, regime, both_antisocial`. `projection_value` is the residual-stream projection onto the trait direction at that layer; `alpha_i/alpha_j` encode the condition (single_a = (1,0), single_b = (0,1), joint = (1,1)). Trajectory layers = 17–32 (16 layers).

**What you compute from these.** The RQ2 mechanism statistics: per-layer projection trajectories `π_i(layer)` under single vs joint steering; the **joint−single divergence `L_div`** vs the noise floor τ (the τ JSON is the split-half bootstrap calibration of measurement noise); whether composition is "mechanically null" (v3, predicted Δπ=0) or shows real interaction. The per-pair parquets are the same data sharded one file per pair for convenient per-pair plotting.

**Pairing rule:** scoring v_k and trajectory v_k are the *same generations* — joint by `pair_id` + condition to cross compositionality outcome (1a) with mechanism (1b).

---

## 2. Supporting datasets Riccardo generated (Phase 13 → 15), NOT for the main stats

These exist in `results/`/`analysis/` and clutter the folder, but they are diagnostic/justification artifacts, not the primary analysis inputs.

### Phase 13 — paper-faithful layer-selection replication (2026-05-05)
Retrospective check that L*=17 reproduces Anthropic's §B.4 L=16 plateau (3 traits × 32 layers, N=5). One-shot validation, no forward dependency.
- `results/layer_selection_paper_repl.json` — aggregate per-(trait,layer) sweep.
- `results/eval_persona_eval_paper_repl/Llama-3.1-8B-Instruct/` — 3 baseline + 96 steer-response CSVs.
- `results/figures/fig_layer_selection_paper_repl.{png,pdf}` — headline figure.

### Phase 14 — judge-calibration audit (2026-05-17)
No new generations. Local audit of v1 CSVs diagnosing six measurement problems. Output = evidence dumps only:
- `analysis/audit_samples/{mid_range,incoherent_high_trait,power_seeking_single}_responses.md`.

### Phase 15 pilots — normalisation-mode selection (2026-05-18 → 22)
The cheap pilots that picked v2's operating point (mode + α). 6 pairs only — **not** the full dataset.
- `results/pilots/composition_normalisations/Llama-3.1-8B-Instruct/` — 66 scored CSVs (6 pairs × 11 settings: False/True/per_axis joints across α∈{4,4.5,5,5.5,6}).
- `results/pilots/composition_normalisations/summary.json` — pilot 1 (False vs True vs per_axis @ α=4).
- `results/pilots/composition_normalisations/pilot2_summary.json` — pilot 2 (True α-sweep + per_axis α=3).
- `results/pilots/composition_normalisations/fig_alpha_sweep_true_{per_pair,aggregate}.{pdf,png}` — dose-response figures + sidecar CSVs.
- `results/alpha_sweep_l17/Llama-3.1-8B-Instruct/*alpha3.0.csv` — 9 single-vector α=3 CSVs (extend E10.3's per-trait α grid; left in place — entangled with the Phase 10 validation α-sweep, not moved).

### Phase 15.16 deep-dive — v3 vs v2 comparison (commit `defa85e`)
The 10-probe exploration + confound audit comparing the per_axis (v3) and normalize=True (v2) runs. These are *derived* comparison tables, regenerable from the v2/v3 datasets:
- `analysis/dataset_compare/` — `phase12.csv`, `phase125.csv`, `v3.csv`, `v125_rowfilt_coh30.csv`, `v3_rowfilt_coh30.csv` + 4 figures (`fig_cos_vs_suppression`, `fig_mechanical_vs_observed`, `fig_per_axis_vs_phase125`, `fig_ratio_topology`).
- `analysis/notebooks/` — `composition_anal.ipynb`, `phase12_vs_phase125_audit.ipynb`, `signed_cosine_predicts_suppression.ipynb`, `per_axis_deep_dive.ipynb`.

---

## 3. `results/` reorganization — APPLIED (composition-only scope)

Done 2026-05-29. The three composition datasets + the normalisation pilots were consolidated into purpose-grouped dirs; the loose root `_summary` / `_tau` / `.parquet` files were folded into the dataset dir they belong to and given clean basenames (`summary.json`, `aggregate.parquet`, `tau.json`). All moves used `git mv` (history preserved); all 63 hard-coded path refs in `scripts/` + notebooks were updated and verified (all 3 datasets + pilot load from new paths, pair counts intact: 36/28/28).

```
results/
├── composition/                              # ← THE THREE ANALYSIS DATASETS
│   ├── v1_phase12_normFalse_a4/
│   │   ├── scoring/{Llama-3.1-8B-Instruct/ (144 CSV), summary.json}
│   │   └── trajectories/{Llama-…/ (36 parquet), aggregate.parquet, tau.json, composition_delta_ldiv.csv}
│   ├── v2_phase125_normTrue_a4.5/            # PRIMARY  (28 pairs)
│   │   ├── scoring/{Llama-…/ (112 CSV), summary.json}
│   │   └── trajectories/{Llama-…/ (28 parquet), aggregate.parquet, tau.json}
│   └── v3_phase1516_perAxis_a4.5/            # mechanical-null  (28 pairs) — same shape
└── pilots/
    └── composition_normalisations/           # 66 CSVs + summary.json + pilot2_summary.json + figs
```

**Intentionally left in place** (chosen scope — `persona_vectors/` alone is read by ~15 scripts; renaming stable infra is high-churn, low-gain):
- `results/persona_vectors/` — the `.pt` steering vectors.
- `results/eval_persona_*`, `results/validation_summary*.json`, `results/logprob_validation_*.json` — extraction + validation outputs.
- `results/layer_selection*.json`, `results/eval_persona_eval_paper_repl/` — Phase 9 + Phase 13.
- `results/alpha_sweep_l17/` (incl. the α=3 CSVs), `results/trajectory_pilot_l17/`, `results/human_eval/`, `results/figures/` — unchanged.

If you later want the full tidy (vectors/, validation/, layer_selection/ grouping), it's a separate ~70-ref pass over the extraction/validation/plotting scripts.
