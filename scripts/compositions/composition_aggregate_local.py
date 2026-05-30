"""
Local aggregation stage for composition scoring - laptop entrypoint.

Single purpose: after composition_judge_local.py has filled every CSV with
trait_a / trait_b / coherence scores, walk all pairs and produce the post-judge
analysis artefacts in one pass:

  - per-pair regime (additive / dominant / suppressive / emergent / mixed)
    from delta_joint / delta_single ratios on judge means
  - per-pair L17 sanity check (closed form pi_a^(1,1)(L*) - pi_a^(1,0)(L*) ~= alpha*cos)
  - results/composition/unnormalized_sum_a4/trajectories/aggregate.parquet - long-form trajectory dataset
    concatenated across per-pair parquets, merged with regime metadata
  - results/composition/unnormalized_sum_a4/trajectories/tau.json - tau via R2 split-half x 1.5
  - results/composition/unnormalized_sum_a4/scoring/summary.json - per-pair regime + delta table + tau

No API calls. No GPU. No HF model load. Pure pandas / numpy.

Run:
    python -m scripts.compositions.composition_aggregate_local
"""
from __future__ import annotations

import json

import pandas as pd

from scripts.compositions.composition_scoring import (
    COMPOSITION_ALPHA,
    HIDDEN_LAYER,
    JUDGE_MODEL,
    LOGS_DIR,
    MODEL_NAME,
    N_PER_QUESTION,
    POLARITY_INVERTED,
    SCORES_OUTPUT_DIR,
    SUMMARY_OUT_PATH,
    TAU_FACTOR,
    TAU_OUT_PATH,
    TAU_Q,
    TAU_RECIPE,
    TRAITS,
    TRAJECTORY_AGG_PARQUET,
    TRAJECTORY_OUT_DIR,
    TRAJECTORY_SETTINGS,
    _baseline_csv_path,
    _calibrate_tau_r2,
    _classify_regime,
    _composition_pairs,
    _csv_has_scores,
    _l17_sanity_check,
    _load_unit_vector_raw,
    _single_csv_path,
    _steered_csv_path,
    _summarise_df,
    _trajectory_pair_parquet,
)
from src.geometry.clusters import pair_cluster_status
from src.geometry.pair_strat import assign_stratum


def _derive_trajectory_layers() -> list[int]:
    """Inspect first per-pair parquet to read the `layer` column. Empty list
    if none exist yet (caller will fall back to `agg['layer'].unique()`)."""
    for cand in TRAJECTORY_OUT_DIR.glob("*.parquet"):
        try:
            return sorted(int(L) for L in pd.read_parquet(cand)["layer"].unique())
        except Exception:
            continue
    return []


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LOCAL AGGREGATE STAGE")
    print(f"  CSV dir     : {SCORES_OUTPUT_DIR}")
    print(f"  parquet dir : {TRAJECTORY_OUT_DIR}")
    print(f"  agg parquet : {TRAJECTORY_AGG_PARQUET}")
    print(f"  τ out       : {TAU_OUT_PATH}")
    print(f"  summary out : {SUMMARY_OUT_PATH}")
    print("  no API, no GPU - pandas/numpy only")
    print("=" * 72 + "\n")

    trajectory_layers = _derive_trajectory_layers()
    if trajectory_layers:
        print(
            f"Trajectory layers (from parquet): "
            f"L ∈ [{trajectory_layers[0]}, {trajectory_layers[-1]}] "
            f"({len(trajectory_layers)} layers)\n"
        )
    else:
        print(
            "WARNING: no per-pair trajectory parquet found in "
            f"{TRAJECTORY_OUT_DIR}. L17 sanity, aggregate parquet, and τ "
            "will be skipped.\n"
        )

    unit_vectors_raw: dict = {}
    for t in TRAITS:
        try:
            unit_vectors_raw[t] = _load_unit_vector_raw(t)
        except FileNotFoundError as e:
            print(f"  WARNING vector: {e}")

    pairs = _composition_pairs()
    print(f"Aggregating {len(pairs)} composition pairs at α={COMPOSITION_ALPHA}\n")

    summary_pairs: list[dict] = []
    pairs_meta: list[dict] = []
    trajectory_frames: list[pd.DataFrame] = []

    for i, (a, b) in enumerate(pairs, 1):
        pair_id = i - 1
        print(f"[{i}/{len(pairs)}] {a} + {b}  (pair_id={pair_id})")

        if a not in unit_vectors_raw or b not in unit_vectors_raw:
            print("  skipping - missing vector")
            summary_pairs.append({"trait_a": a, "trait_b": b, "status": "MISSING_VEC"})
            continue

        v_a_raw, v_b_raw = unit_vectors_raw[a], unit_vectors_raw[b]
        cos_ab = float((v_a_raw @ v_b_raw).item())

        base_csv = _baseline_csv_path(a, b)
        sa_csv = _single_csv_path(a, b, "a", COMPOSITION_ALPHA)
        sb_csv = _single_csv_path(a, b, "b", COMPOSITION_ALPHA)
        joint_csv = _steered_csv_path(a, b, COMPOSITION_ALPHA)
        if not all(p.exists() for p in (base_csv, sa_csv, sb_csv, joint_csv)):
            print("  skipping - one or more CSVs missing")
            summary_pairs.append({"trait_a": a, "trait_b": b, "status": "MISSING_CSV"})
            continue
        if not all(_csv_has_scores(p) for p in (base_csv, sa_csv, sb_csv, joint_csv)):
            print("  skipping - one or more CSVs lack judge scores (run composition_judge_local.py first)")
            summary_pairs.append({"trait_a": a, "trait_b": b, "status": "UNSCORED_CSV"})
            continue

        base = _summarise_df(pd.read_csv(base_csv))
        single_a = _summarise_df(pd.read_csv(sa_csv))
        single_b = _summarise_df(pd.read_csv(sb_csv))
        steered = _summarise_df(pd.read_csv(joint_csv))

        delta_a_joint = steered["trait_a_mean"] - base["trait_a_mean"]
        delta_b_joint = steered["trait_b_mean"] - base["trait_b_mean"]
        delta_a_single = single_a["trait_a_mean"] - base["trait_a_mean"]
        delta_b_single = single_b["trait_b_mean"] - base["trait_b_mean"]
        delta_comp = steered["composition_mean"] - base["composition_mean"]
        delta_coh = steered["coherence_mean"] - base["coherence_mean"]
        regime = _classify_regime(
            delta_a_joint, delta_b_joint, delta_a_single, delta_b_single,
        )

        print(
            f"  base comp={base['composition_mean']:.2f}  "
            f"joint comp={steered['composition_mean']:.2f} "
            f"(Δ{delta_comp:+.2f})  regime={regime}  cos={cos_ab:+.3f}"
        )

        sanity = None
        df_traj = None
        traj_path = _trajectory_pair_parquet(a, b)
        if traj_path.exists():
            df_traj = pd.read_parquet(traj_path)
            sanity = _l17_sanity_check(df_traj, a, b, COMPOSITION_ALPHA, cos_ab)
            print(
                f"  L={sanity['layer']} sanity: pred[{sanity['scheme']}]={sanity['pred_diff']:+.3f}  "
                f"obs Δπ_a={sanity['obs_pi_a_diff']:+.3f}  "
                f"obs Δπ_b={sanity['obs_pi_b_diff']:+.3f}"
            )
            trajectory_frames.append(df_traj)
        else:
            print(f"  no per-pair parquet at {traj_path}; skipping L17 sanity")

        cluster_status = pair_cluster_status(a, b)
        pairs_meta.append({
            "pair_id": pair_id,
            "trait_i": a,
            "trait_j": b,
            "cosine": cos_ab,
            "stratum": assign_stratum(abs(cos_ab)),
            "regime": regime,
            "both_antisocial": cluster_status == "within_antisocial",
        })

        entry: dict = {
            "trait_a": a,
            "trait_b": b,
            "status": "ok",
            "alpha": COMPOSITION_ALPHA,
            "cos": round(cos_ab, 4),
            "regime": regime,
            "baseline": {k: round(v, 2) for k, v in base.items()},
            "single_a": {k: round(v, 2) for k, v in single_a.items()},
            "single_b": {k: round(v, 2) for k, v in single_b.items()},
            "steered": {k: round(v, 2) for k, v in steered.items()},
            "delta": {
                "trait_a_joint": round(delta_a_joint, 2),
                "trait_b_joint": round(delta_b_joint, 2),
                "trait_a_single": round(delta_a_single, 2),
                "trait_b_single": round(delta_b_single, 2),
                "composition": round(delta_comp, 2),
                "coherence": round(delta_coh, 2),
            },
        }
        if sanity is not None:
            entry["l17_sanity"] = {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in sanity.items()
            }
        summary_pairs.append(entry)

        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "judge_model": JUDGE_MODEL,
                "layer": HIDDEN_LAYER,
                "alpha": COMPOSITION_ALPHA,
                "vector_normalisation": "unit",
                "injection": "h += alpha * (v_a_unit + v_b_unit)",
                "polarity_inverted_for_steering": sorted(POLARITY_INVERTED),
                "projection_axis": "raw_direction_no_polarity_flip",
                "n_per_question": N_PER_QUESTION,
                "trajectory_layers": trajectory_layers,
                "trajectory_settings": [list(s) for s in TRAJECTORY_SETTINGS],
                "tau_recipe": TAU_RECIPE,
                "tau_value": None,
                "pairs": summary_pairs,
            }, f, indent=2)

    # End-of-stage tally.
    n_csvs = len(list(SCORES_OUTPUT_DIR.glob("*.csv")))
    n_scored = sum(1 for p in SCORES_OUTPUT_DIR.glob("*.csv") if _csv_has_scores(p))
    n_parquets = len(list(TRAJECTORY_OUT_DIR.glob("*.parquet")))
    n_ok = sum(1 for r in summary_pairs if r["status"] == "ok")
    print()
    print("=" * 72)
    print("AGGREGATE STAGE TALLY")
    print(f"  CSVs on disk          : {n_csvs}  (expected 144)")
    print(f"  CSVs with judge scores: {n_scored}  (target 144)")
    print(f"  Per-pair parquets     : {n_parquets}  (expected 36)")
    print(f"  Pairs fully aggregated: {n_ok}  (target 36)")
    print("=" * 72)

    if trajectory_frames and pairs_meta:
        agg = pd.concat(trajectory_frames, ignore_index=True)
        agg = agg.merge(pd.DataFrame(pairs_meta), on="pair_id", how="left")
        TRAJECTORY_AGG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(TRAJECTORY_AGG_PARQUET, index=False)
        print(
            f"\nAggregate parquet: {len(agg):,} rows  "
            f"{agg['pair_id'].nunique()} pairs -> {TRAJECTORY_AGG_PARQUET}"
        )

        layers_for_tau = trajectory_layers or sorted(int(L) for L in agg["layer"].unique())
        print(f"Calibrating τ via {TAU_RECIPE} ...")
        tau_summary = _calibrate_tau_r2(agg, layers_for_tau, COMPOSITION_ALPHA)
        TAU_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAU_OUT_PATH.write_text(json.dumps(tau_summary, indent=2))
        print(
            f"  τ = {tau_summary['tau']:.4f}  "
            f"(q{int(TAU_Q * 100)}={tau_summary.get('q_value_pre_factor', float('nan')):.4f} × {TAU_FACTOR}, "
            f"n_groups={tau_summary['n_groups']}, "
            f"n_draws={tau_summary['n_draws_total']:,})  -> {TAU_OUT_PATH}"
        )

        with open(SUMMARY_OUT_PATH) as f:
            payload = json.load(f)
        payload["tau_value"] = tau_summary["tau"]
        payload["tau_summary_path"] = str(TAU_OUT_PATH)
        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump(payload, f, indent=2)

    print(f"\nSummary -> {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
