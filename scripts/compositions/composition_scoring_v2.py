"""
Phase 12.5 — recalibrated joint composition scoring (Riccardo, 2026-05-22+).

Successor to scripts/compositions/composition_scoring.py (Phase 12 / Edoardo).
Same pipeline shape, three changes relative to Phase 12 — all locked in
paper/experiments_log.md §E15.13:

  1. Joint composition normalisation: False -> True
     Phase 12: h += α · (v̂_a + v̂_b)        (‖δ‖ varies 4.0–7.4 with cosine)
     Phase 12.5: h += α · (v̂_a + v̂_b) / ‖v̂_a + v̂_b‖   (‖δ‖ = α constant)

  2. α: 4.0 -> 4.5
     Picked from the Phase 15 pilot dose-response: utility = Δcomp × coh / 100
     peaks at α=4.5 (coh=69 ± 6, comfortable margin above the Anthropic
     coh=50 threshold). See [results/pilots/composition_normalisations/
     fig_alpha_sweep_true_aggregate.png] and E15.11/E15.13.

  3. Drop `power_seeking` from the main trait set: 9 -> 8 traits, 36 -> 28
     pairs. Phase 14 Problem 1 + Phase 15 pilot pair 6 both confirmed the
     trait × judge × prompt interaction is broken on these prompts in every
     normalisation mode.

The Phase 12 outputs (results/composition/v1_phase12_normFalse_a4/scoring/,
.../trajectories/aggregate.parquet, ...) are preserved untouched. Phase 12.5
lands under the v2 dataset dir:
    results/composition/v2_phase125_normTrue_a4.5/scoring/Llama-3.1-8B-Instruct/
    results/composition/v2_phase125_normTrue_a4.5/scoring/summary.json
    results/composition/v2_phase125_normTrue_a4.5/trajectories/Llama-3.1-8B-Instruct/
    results/composition/v2_phase125_normTrue_a4.5/trajectories/aggregate.parquet
    results/composition/v2_phase125_normTrue_a4.5/trajectories/tau.json

Run (same 3-stage mode-gated pattern as Phase 12):
    # stage 1 (compute / GPU, no internet): generate completions + trajectories
    COMPOSITION_MODE=generate python -m scripts.compositions.composition_scoring_v2
    # stage 2 (login / no GPU, internet):   judge CSVs + aggregate + τ + summary
    COMPOSITION_MODE=judge    python -m scripts.compositions.composition_scoring_v2
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.composition.joint_injection import (
    compose_steering_vector,
    project_activation,
    trajectory_response_avg,
)
from src.extraction.generation import COHERENCE_PROMPT, _judge_all, generate_batch
from src.geometry.clusters import pair_cluster_status
from src.geometry.pair_strat import assign_stratum
from src.inference.hf_model import load_hf_model
from src.judge import OpenAiJudge

load_dotenv()

# ---------------------------------------------------------------------------
# Config — Phase 12.5 locked operating point (E15.13)
# ---------------------------------------------------------------------------
# 8 traits (power_seeking dropped from Phase 12's 9-trait set).
TRAITS = [
    # Tier S
    "apathetic",
    "evil",
    "hallucinating",
    "humorous",
    "impolite",
    "sycophantic",
    # Tier A (without power_seeking)
    "confidence",
    "formality",
]

# No polarity inversions remain — the only Phase 12 inversion was power_seeking.
POLARITY_INVERTED: set[str] = set()

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"

HIDDEN_LAYER = 17
HOOK_LAYER_IDX = HIDDEN_LAYER - 1

# Phase 12.5 operating point.
COMPOSITION_ALPHA = 4.5
COMPOSITION_NORMALIZE: bool | str = True   # was False in Phase 12

# LLM-judge stage settings — held identical to Phase 12 / E7.8 / E9.7 / alpha-sweep.
N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
MAX_CONCURRENT_JUDGES = 5

VECTOR_OUTPUT_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct")
COMPOSITION_DATA_DIR = Path("data/composition_eval")
SCORES_OUTPUT_DIR = Path("results/composition/v2_phase125_normTrue_a4.5/scoring/Llama-3.1-8B-Instruct")
SUMMARY_OUT_PATH = Path("results/composition/v2_phase125_normTrue_a4.5/scoring/summary.json")
LOGS_DIR = Path("logs")

# --- Phase 2 trajectory dataset (same protocol as Phase 12, v2 paths) -------
TRAJECTORY_SETTINGS: list[tuple[int, int]] = [(1, 0), (0, 1), (1, 1)]
TRAJECTORY_OUT_DIR = Path("results/composition/v2_phase125_normTrue_a4.5/trajectories/Llama-3.1-8B-Instruct")
TRAJECTORY_AGG_PARQUET = Path("results/composition/v2_phase125_normTrue_a4.5/trajectories/aggregate.parquet")

# Regime classification thresholds — held identical to Phase 12 for direct
# comparability of the per-pair regime labels.
REGIME_ADDITIVE_LO = 0.7
REGIME_ADDITIVE_HI = 1.3
REGIME_DOMINANT_LO = 0.7
REGIME_DOMINANT_HI = 0.3
REGIME_SUPPRESSIVE_MAX = 0.5
REGIME_EMERGENT_MIN = 1.3
# ---------------------------------------------------------------------------


# === IO helpers ============================================================

def _composition_pairs() -> list[tuple[str, str]]:
    """All unordered (a, b) with a < b over TRAITS — 28 for the 8-trait set."""
    return list(combinations(sorted(TRAITS), 2))


def _load_composition_artifact(trait_a: str, trait_b: str) -> dict:
    a, b = sorted([trait_a, trait_b])
    path = COMPOSITION_DATA_DIR / f"{a}__{b}.json"
    return json.loads(path.read_text())


def _load_unit_vector(trait: str) -> torch.Tensor:
    """Unit-normalised vector at L=17 with the alpha-sweep polarity flip.
    For the 8-trait Phase 12.5 set, POLARITY_INVERTED is empty (the only
    Phase 12 inversion was power_seeking, now dropped), so this just
    unit-normalises. Kept symmetric with Phase 12 for code clarity.
    """
    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector not found at {vec_path}")
    stack = torch.load(vec_path, map_location="cpu")
    v = stack[HIDDEN_LAYER]
    if trait in POLARITY_INVERTED:
        v = -v
    return v / v.norm()


def _load_unit_vector_raw(trait: str) -> torch.Tensor:
    """Raw-direction unit vector (no polarity flip) — projection axis for
    trajectories. With POLARITY_INVERTED empty in Phase 12.5, this is
    identical to _load_unit_vector but kept separate for symmetry with the
    Phase 12 trajectory-capture convention.
    """
    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector not found at {vec_path}")
    stack = torch.load(vec_path, map_location="cpu")
    v = stack[HIDDEN_LAYER]
    return v / v.norm()


def _baseline_csv_path(trait_a: str, trait_b: str) -> Path:
    a, b = sorted([trait_a, trait_b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_baseline.csv"


def _steered_csv_path(trait_a: str, trait_b: str, alpha: float) -> Path:
    a, b = sorted([trait_a, trait_b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_joint_alpha{alpha}.csv"


def _single_csv_path(trait_a: str, trait_b: str, which: str, alpha: float) -> Path:
    a, b = sorted([trait_a, trait_b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_single_{which}_alpha{alpha}.csv"


def _trajectory_pair_parquet(trait_a: str, trait_b: str) -> Path:
    a, b = sorted([trait_a, trait_b])
    return TRAJECTORY_OUT_DIR / f"{a}__{b}.parquet"


# === LLM helpers ===========================================================

def _build_eval_conversations(questions: list[str], n_per_question: int):
    convs, questions_flat = [], []
    for q in questions:
        for _ in range(n_per_question):
            convs.append([{"role": "user", "content": q}])
            questions_flat.append(q)
    return convs, questions_flat


def _judge_run_composition(
    judge_model: str,
    eval_prompt_a: str,
    eval_prompt_b: str,
    questions: list[str],
    answers: list[str],
    max_concurrent: int,
    progress_tag: str = "",
) -> tuple[list, list, list]:
    judge_a = OpenAiJudge(judge_model, eval_prompt_a, eval_type="0_100")
    judge_b = OpenAiJudge(judge_model, eval_prompt_b, eval_type="0_100")
    judge_coh = OpenAiJudge(judge_model, COHERENCE_PROMPT, eval_type="0_100")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores_a = loop.run_until_complete(_judge_all(
            judge_a, questions, answers, max_concurrent,
            progress_label=f"{progress_tag} trait_a", progress_every=25,
        ))
        scores_b = loop.run_until_complete(_judge_all(
            judge_b, questions, answers, max_concurrent,
            progress_label=f"{progress_tag} trait_b", progress_every=25,
        ))
        scores_coh = loop.run_until_complete(_judge_all(
            judge_coh, questions, answers, max_concurrent,
            progress_label=f"{progress_tag} coherence", progress_every=25,
        ))
    finally:
        loop.close()
    return scores_a, scores_b, scores_coh


def _mean(col) -> float:
    valid = [x for x in col if x is not None and pd.notna(x)]
    return sum(valid) / len(valid) if valid else float("nan")


def _summarise_df(df: pd.DataFrame) -> dict:
    return {
        "trait_a_mean": _mean(df["trait_a"]),
        "trait_b_mean": _mean(df["trait_b"]),
        "composition_mean": _mean(df["composition"]),
        "coherence_mean": _mean(df["coherence"]),
    }


# === Stage runners =========================================================

def _csv_has_scores(p: Path) -> bool:
    if not p.exists():
        return False
    df = pd.read_csv(p)
    return "trait_a" in df.columns and df["trait_a"].notna().any()


def _csv_status(p: Path) -> str:
    if not p.exists():
        return "no CSV"
    df = pd.read_csv(p)
    n = len(df)
    scored = int(df["trait_a"].notna().sum()) if "trait_a" in df.columns else 0
    return f"{n} rows, {scored} scored"


def _generate_completions_csv(
    out_csv: Path,
    model, tok,
    artifact: dict,
    steering,
    log_path: Path,
    header: str,
) -> None:
    if out_csv.exists():
        return
    convs, questions_flat = _build_eval_conversations(artifact["questions"], N_PER_QUESTION)
    with log_path.open("w", buffering=1) as fh, redirect_stdout(fh), redirect_stderr(fh):
        print(header)
        _, answers = generate_batch(
            model, tok, convs,
            max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, batch_size=BATCH_SIZE,
            steering=steering,
        )
    df = pd.DataFrame({
        "question": questions_flat,
        "answer": answers,
        "trait_a": pd.array([pd.NA] * len(answers), dtype="Float64"),
        "trait_b": pd.array([pd.NA] * len(answers), dtype="Float64"),
        "coherence": pd.array([pd.NA] * len(answers), dtype="Float64"),
        "composition": pd.array([pd.NA] * len(answers), dtype="Float64"),
    })
    df.to_csv(out_csv, index=False)


def _judge_csv_inplace(
    out_csv: Path,
    eval_prompt_a: str,
    eval_prompt_b: str,
    progress_tag: str,
    log_path: Path,
) -> None:
    if not out_csv.exists():
        return
    df = pd.read_csv(out_csv)
    if "trait_a" not in df.columns:
        return
    mask = df["trait_a"].isna()
    if not mask.any():
        return
    questions = df.loc[mask, "question"].astype(str).tolist()
    answers = df.loc[mask, "answer"].astype(str).fillna("").tolist()
    with log_path.open("a", buffering=1) as fh, redirect_stdout(fh), redirect_stderr(fh):
        print(f"[judge stage] {progress_tag}  rows_to_judge={len(questions)}")
        scores_a, scores_b, scores_coh = _judge_run_composition(
            JUDGE_MODEL,
            eval_prompt_a, eval_prompt_b,
            questions, answers, MAX_CONCURRENT_JUDGES,
            progress_tag=progress_tag,
        )

    def _to_nan(xs):
        return [float("nan") if x is None else float(x) for x in xs]

    df.loc[mask, "trait_a"] = _to_nan(scores_a)
    df.loc[mask, "trait_b"] = _to_nan(scores_b)
    df.loc[mask, "coherence"] = _to_nan(scores_coh)
    df["composition"] = df[["trait_a", "trait_b"]].mean(axis=1)
    df.to_csv(out_csv, index=False)


def _run_baseline_composition(
    trait_a: str, trait_b: str, artifact: dict, model, tok, log_path: Path, mode: str,
) -> dict | None:
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _baseline_csv_path(trait_a, trait_b)
    n_q = N_PER_QUESTION * len(artifact["questions"])
    if mode in ("generate", "full"):
        _generate_completions_csv(
            out_csv, model, tok, artifact,
            steering=None, log_path=log_path,
            header=f"pair={trait_a}+{trait_b}  baseline  questions={n_q}",
        )
    if mode in ("judge", "full"):
        _judge_csv_inplace(
            out_csv,
            artifact["eval_prompt_a"], artifact["eval_prompt_b"],
            progress_tag=f"{trait_a}+{trait_b} baseline",
            log_path=log_path,
        )
    if _csv_has_scores(out_csv):
        return _summarise_df(pd.read_csv(out_csv))
    return None


def _run_joint_steered_composition(
    trait_a: str, trait_b: str, artifact: dict, alpha: float,
    model, tok, v_a_unit: torch.Tensor, v_b_unit: torch.Tensor, log_path: Path,
    mode: str,
) -> dict | None:
    """Phase 12.5 joint condition. Builds δ via compose_steering_vector with
    `normalize=COMPOSITION_NORMALIZE` (True): δ = α · (v̂_a + v̂_b) / ‖v̂_a + v̂_b‖,
    so ‖δ‖ = α regardless of pair geometry — the key recalibration vs Phase 12.
    """
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _steered_csv_path(trait_a, trait_b, alpha)
    n_q = N_PER_QUESTION * len(artifact["questions"])
    if mode in ("generate", "full"):
        delta = compose_steering_vector(
            [(v_a_unit, 1.0), (v_b_unit, 1.0)],
            alpha=alpha, normalize=COMPOSITION_NORMALIZE,
        )
        delta_norm = float(delta.norm().item())
        _generate_completions_csv(
            out_csv, model, tok, artifact,
            steering=(delta, HOOK_LAYER_IDX, 1.0, "response"),
            log_path=log_path,
            header=(
                f"pair={trait_a}+{trait_b}  joint normalize=True α={alpha}  "
                f"layer={HIDDEN_LAYER} (hook block {HOOK_LAYER_IDX})  "
                f"‖δ‖={delta_norm:.3f}  questions={n_q}"
            ),
        )
    if mode in ("judge", "full"):
        _judge_csv_inplace(
            out_csv,
            artifact["eval_prompt_a"], artifact["eval_prompt_b"],
            progress_tag=f"{trait_a}+{trait_b} joint α={alpha}",
            log_path=log_path,
        )
    if _csv_has_scores(out_csv):
        return _summarise_df(pd.read_csv(out_csv))
    return None


def _run_single_steered_composition(
    trait_a: str, trait_b: str, w_a: int, w_b: int, artifact: dict, alpha: float,
    model, tok, v_a_unit: torch.Tensor, v_b_unit: torch.Tensor, log_path: Path,
    mode: str,
) -> dict | None:
    """Single-vector setting (1,0) or (0,1). With one weight = 0, every
    `normalize` mode collapses to α·v̂_active, so this is identical to Phase
    12's single condition regardless of the COMPOSITION_NORMALIZE setting.
    Pass-through to compose_steering_vector with the same kwarg as the joint
    runner for code symmetry.
    """
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    which = "a" if (w_a, w_b) == (1, 0) else "b"
    out_csv = _single_csv_path(trait_a, trait_b, which, alpha)
    n_q = N_PER_QUESTION * len(artifact["questions"])
    if mode in ("generate", "full"):
        delta = compose_steering_vector(
            [(v_a_unit, float(w_a)), (v_b_unit, float(w_b))],
            alpha=alpha, normalize=COMPOSITION_NORMALIZE,
        )
        _generate_completions_csv(
            out_csv, model, tok, artifact,
            steering=(delta, HOOK_LAYER_IDX, 1.0, "response"),
            log_path=log_path,
            header=(
                f"pair={trait_a}+{trait_b}  single ({w_a},{w_b}) α={alpha}  "
                f"layer={HIDDEN_LAYER}  questions={n_q}"
            ),
        )
    if mode in ("judge", "full"):
        _judge_csv_inplace(
            out_csv,
            artifact["eval_prompt_a"], artifact["eval_prompt_b"],
            progress_tag=f"{trait_a}+{trait_b} single_{which} α={alpha}",
            log_path=log_path,
        )
    if _csv_has_scores(out_csv):
        return _summarise_df(pd.read_csv(out_csv))
    return None


def _classify_regime(
    delta_a_joint: float, delta_b_joint: float,
    delta_a_single: float, delta_b_single: float,
) -> str:
    eps = 1e-3
    ratio_a = delta_a_joint / delta_a_single if abs(delta_a_single) > eps else 0.0
    ratio_b = delta_b_joint / delta_b_single if abs(delta_b_single) > eps else 0.0

    if (REGIME_ADDITIVE_LO <= ratio_a <= REGIME_ADDITIVE_HI
            and REGIME_ADDITIVE_LO <= ratio_b <= REGIME_ADDITIVE_HI):
        return "additive"
    if ratio_a < REGIME_SUPPRESSIVE_MAX and ratio_b < REGIME_SUPPRESSIVE_MAX:
        return "suppressive"
    if ratio_a > REGIME_EMERGENT_MIN and ratio_b > REGIME_EMERGENT_MIN:
        return "emergent"
    if ((ratio_a >= REGIME_DOMINANT_LO and ratio_b <= REGIME_DOMINANT_HI)
            or (ratio_b >= REGIME_DOMINANT_LO and ratio_a <= REGIME_DOMINANT_HI)):
        return "dominant"
    return "mixed"


def _capture_trajectories_for_pair(
    pair_id: int, trait_a: str, trait_b: str,
    model, tok,
    v_a_steer: torch.Tensor, v_b_steer: torch.Tensor,
    v_a_proj: torch.Tensor, v_b_proj: torch.Tensor,
    layers: list[int], alpha: float,
) -> pd.DataFrame:
    """Trajectory capture mirrors Phase 12 — same protocol, same per-prompt
    teacher-forcing — but uses `normalize=COMPOSITION_NORMALIZE` so the
    injected δ matches what generation actually saw."""
    out_path = _trajectory_pair_parquet(trait_a, trait_b)
    if out_path.exists():
        return pd.read_parquet(out_path)

    csv_per_setting = {
        (1, 0): _single_csv_path(trait_a, trait_b, "a", alpha),
        (0, 1): _single_csv_path(trait_a, trait_b, "b", alpha),
        (1, 1): _steered_csv_path(trait_a, trait_b, alpha),
    }
    for setting, p in csv_per_setting.items():
        if not p.exists():
            raise FileNotFoundError(
                f"trajectory capture needs CSV for setting {setting}: {p}"
            )

    rows: list[dict] = []
    for setting, csv_path in csv_per_setting.items():
        df = pd.read_csv(csv_path)
        w_a, w_b = setting
        delta = compose_steering_vector(
            [(v_a_steer, float(w_a)), (v_b_steer, float(w_b))],
            alpha=alpha, normalize=COMPOSITION_NORMALIZE,
        )
        for prompt_id, (q, ans) in enumerate(zip(df["question"], df["answer"])):
            question_id = prompt_id // N_PER_QUESTION
            completion_id = prompt_id % N_PER_QUESTION
            avg = trajectory_response_avg(
                model, tok, str(q), str(ans),
                layers_above=layers,
                delta_at_lstar=delta,
                layer_star=HIDDEN_LAYER,
            )
            for L in layers:
                h = avg[L]
                pi_a = float(project_activation(h, v_a_proj).item())
                pi_b = float(project_activation(h, v_b_proj).item())
                rows.append({
                    "pair_id": pair_id, "behaviour_index": 0,
                    "alpha_i": float(w_a) * alpha, "alpha_j": float(w_b) * alpha,
                    "prompt_id": prompt_id,
                    "question_id": question_id, "completion_id": completion_id,
                    "layer": int(L),
                    "projection_value": pi_a,
                })
                rows.append({
                    "pair_id": pair_id, "behaviour_index": 1,
                    "alpha_i": float(w_a) * alpha, "alpha_j": float(w_b) * alpha,
                    "prompt_id": prompt_id,
                    "question_id": question_id, "completion_id": completion_id,
                    "layer": int(L),
                    "projection_value": pi_b,
                })
    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    return df_out


# === τ R2 calibration (Phase 2 pre-registered) =============================

TAU_RECIPE = "R2_split_half_bootstrap_q95_x1.5"
TAU_FACTOR = 1.5
TAU_BOOTSTRAP_DRAWS = 1000
TAU_Q = 0.95
TAU_OUT_PATH = Path("results/composition/v2_phase125_normTrue_a4.5/trajectories/tau.json")


def _calibrate_tau_r2(
    agg_df: pd.DataFrame,
    layers: list[int],
    alpha: float,
    seed: int = 42,
) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)

    indiv_mask = (
        ((agg_df["alpha_i"] == alpha) & (agg_df["alpha_j"] == 0.0)
         & (agg_df["behaviour_index"] == 0))
        | ((agg_df["alpha_i"] == 0.0) & (agg_df["alpha_j"] == alpha)
           & (agg_df["behaviour_index"] == 1))
    )
    sub = agg_df.loc[indiv_mask]

    pooled_max_diffs: list[float] = []
    group_cols = ["pair_id", "alpha_i", "alpha_j", "behaviour_index"]
    for _, g in sub.groupby(group_cols, sort=False):
        wide = g.pivot_table(
            index="prompt_id", columns="layer", values="projection_value", aggfunc="mean"
        )
        wide = wide.reindex(columns=layers)
        arr = wide.to_numpy(dtype=float)
        n = arr.shape[0]
        if n < 2:
            continue
        half = n // 2
        for _ in range(TAU_BOOTSTRAP_DRAWS):
            idx = rng.permutation(n)
            a_mean = arr[idx[:half]].mean(axis=0)
            b_mean = arr[idx[half:half * 2]].mean(axis=0)
            pooled_max_diffs.append(float(np.max(np.abs(a_mean - b_mean))))

    if not pooled_max_diffs:
        return {
            "recipe": TAU_RECIPE, "tau": float("nan"),
            "n_groups": 0, "n_draws_total": 0,
            "factor": TAU_FACTOR, "q": TAU_Q, "bootstrap_draws_per_group": TAU_BOOTSTRAP_DRAWS,
            "note": "no individual-steering rows found",
        }
    arr = np.array(pooled_max_diffs)
    q_val = float(np.quantile(arr, TAU_Q))
    tau = TAU_FACTOR * q_val
    return {
        "recipe": TAU_RECIPE, "tau": tau,
        "factor": TAU_FACTOR, "q": TAU_Q, "q_value_pre_factor": q_val,
        "bootstrap_draws_per_group": TAU_BOOTSTRAP_DRAWS,
        "n_groups": int(sub.groupby(group_cols).ngroups),
        "n_draws_total": int(arr.size),
        "pooled_diff_min": float(arr.min()),
        "pooled_diff_max": float(arr.max()),
        "pooled_diff_mean": float(arr.mean()),
    }


def _l17_sanity_check(
    df_pair: pd.DataFrame, trait_a: str, trait_b: str,
    alpha: float, cos_ij: float,
) -> dict:
    """Closed-form sanity at L*=17. Under Phase 12.5's normalize=True:
        δ = α · (v̂_a + v̂_b) / ‖v̂_a + v̂_b‖
        ⟨δ, v̂_a⟩ = α · (1+cos) / √(2+2cos) = α · √((1+cos)/2)
    The Phase 12 prediction π_a^(1,1) − π_a^(1,0) ≈ α·cos no longer holds —
    the new prediction is α · (√((1+cos)/2) − 1). Both values are reported so
    the figure shows the recalibration explicitly.
    """
    import math
    L = HIDDEN_LAYER
    sub = df_pair[df_pair["layer"] == L]

    def _mean_at(behaviour_index, ai, aj):
        m = sub[(sub["behaviour_index"] == behaviour_index)
                & (sub["alpha_i"] == ai) & (sub["alpha_j"] == aj)]
        return float(m["projection_value"].mean()) if len(m) else float("nan")

    pi_a_10 = _mean_at(0, alpha, 0.0)
    pi_a_11 = _mean_at(0, alpha, alpha)
    pi_b_01 = _mean_at(1, 0.0, alpha)
    pi_b_11 = _mean_at(1, alpha, alpha)

    one_plus_cos = max(1.0 + cos_ij, 1e-9)
    pred_normalize_true = alpha * (math.sqrt(one_plus_cos / 2.0) - 1.0)
    pred_phase12 = alpha * cos_ij   # reported for reference only — not the v2 prediction
    return {
        "layer": L, "alpha": alpha, "cos": cos_ij,
        "pred_diff_normalize_true": pred_normalize_true,
        "pred_diff_phase12_false": pred_phase12,
        "obs_pi_a_diff": pi_a_11 - pi_a_10,
        "obs_pi_b_diff": pi_b_11 - pi_b_01,
    }


# === Orchestration =========================================================

def main() -> None:
    mode = os.environ.get("COMPOSITION_MODE", "full").lower()
    if mode not in {"generate", "judge", "full"}:
        raise SystemExit(
            f"COMPOSITION_MODE must be one of generate|judge|full, got {mode!r}"
        )
    print(f"COMPOSITION_MODE={mode}  (Phase 12.5: normalize=True, α={COMPOSITION_ALPHA})")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = tok = None
    if mode != "judge":
        print(f"Loading {MODEL_NAME} ...")
        model, tok = load_hf_model(MODEL_NAME)
        print(
            f"Model loaded: hidden={model.config.hidden_size}  "
            f"n_layers={model.config.num_hidden_layers}\n"
        )
        n_layers = model.config.num_hidden_layers
        trajectory_layers = list(range(HIDDEN_LAYER, n_layers + 1))
    else:
        trajectory_layers = []
        for cand in TRAJECTORY_OUT_DIR.glob("*.parquet"):
            try:
                trajectory_layers = sorted(int(L) for L in pd.read_parquet(cand)["layer"].unique())
                break
            except Exception:
                continue
        print("Judge mode: skipping HF model load.")
    if trajectory_layers:
        print(
            f"Trajectory layers: L ∈ [{trajectory_layers[0]}, {trajectory_layers[-1]}] "
            f"({len(trajectory_layers)} layers)"
        )

    unit_vectors: dict[str, torch.Tensor] = {}
    unit_vectors_raw: dict[str, torch.Tensor] = {}
    for t in TRAITS:
        try:
            unit_vectors[t] = _load_unit_vector(t)
            unit_vectors_raw[t] = _load_unit_vector_raw(t)
            inv = " (inverted)" if t in POLARITY_INVERTED else ""
            print(f"  loaded unit vector for {t}{inv}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    pairs = _composition_pairs()
    n_pairs = len(pairs)
    print(f"\nEvaluating {n_pairs} composition pairs at α={COMPOSITION_ALPHA}, normalize=True")
    print("=" * 72)
    if mode == "generate":
        print(
            f"GENERATE STAGE — for each of {n_pairs} pairs, produce 4 CSVs of LLM completions\n"
            "  (baseline, single_a, single_b, joint) + 1 trajectory parquet.\n"
            f"  CSVs written under   {SCORES_OUTPUT_DIR}\n"
            f"  parquets written to  {TRAJECTORY_OUT_DIR}\n"
            f"  per-pair logs at     {LOGS_DIR}/composition_<a>__<b>_*.log\n"
            "  CSV score columns left as NaN — judge stage runs off-cluster.\n"
            "  Tail progress:  tail -F logs/composition_<pair>_*.log"
        )
    elif mode == "judge":
        print(
            f"JUDGE STAGE — for each of {n_pairs} pairs, read 4 CSVs from disk and fill\n"
            "  NaN score columns by calling OpenAI judge (trait_a, trait_b, coherence).\n"
            f"  CSVs read+written under  {SCORES_OUTPUT_DIR}\n"
            f"  per-pair logs at         {LOGS_DIR}/composition_<a>__<b>_*.log\n"
            "  [judge] done/total progress lines emitted to this stdout every 25 rows."
        )
    else:
        print(
            "FULL STAGE — generate completions then judge them in one process.\n"
            "  Needs both GPU and outbound internet."
        )
    print("=" * 72 + "\n")

    summary_pairs: list[dict] = []
    pairs_meta: list[dict] = []
    trajectory_frames: list[pd.DataFrame] = []

    for i, (a, b) in enumerate(pairs, 1):
        pair_id = i - 1
        print(f"\n[{i}/{n_pairs}] {a} + {b}  (pair_id={pair_id})")

        if a not in unit_vectors or b not in unit_vectors:
            print(f"  skipping — missing vector")
            summary_pairs.append({"trait_a": a, "trait_b": b, "status": "MISSING_VEC"})
            continue
        try:
            artifact = _load_composition_artifact(a, b)
        except FileNotFoundError as e:
            print(f"  skipping — {e}")
            summary_pairs.append({"trait_a": a, "trait_b": b, "status": "MISSING_ARTIFACT"})
            continue

        v_a, v_b = unit_vectors[a], unit_vectors[b]
        v_a_raw, v_b_raw = unit_vectors_raw[a], unit_vectors_raw[b]
        cos_ab = float((v_a_raw @ v_b_raw).item())

        base_log = LOGS_DIR / f"composition_v2_{a}__{b}_baseline.log"
        base_csv = _baseline_csv_path(a, b)
        print(f"  baseline                  -> {base_csv.name}")
        base = _run_baseline_composition(a, b, artifact, model, tok, base_log, mode)
        print(f"    {_csv_status(base_csv)}    log={base_log.name}")

        single_a_log = LOGS_DIR / f"composition_v2_{a}__{b}_single_a_alpha{COMPOSITION_ALPHA}.log"
        single_a_csv = _single_csv_path(a, b, "a", COMPOSITION_ALPHA)
        print(f"  single_a (1,0) α={COMPOSITION_ALPHA}    -> {single_a_csv.name}")
        single_a = _run_single_steered_composition(
            a, b, 1, 0, artifact, COMPOSITION_ALPHA, model, tok, v_a, v_b, single_a_log, mode,
        )
        print(f"    {_csv_status(single_a_csv)}    log={single_a_log.name}")

        single_b_log = LOGS_DIR / f"composition_v2_{a}__{b}_single_b_alpha{COMPOSITION_ALPHA}.log"
        single_b_csv = _single_csv_path(a, b, "b", COMPOSITION_ALPHA)
        print(f"  single_b (0,1) α={COMPOSITION_ALPHA}    -> {single_b_csv.name}")
        single_b = _run_single_steered_composition(
            a, b, 0, 1, artifact, COMPOSITION_ALPHA, model, tok, v_a, v_b, single_b_log, mode,
        )
        print(f"    {_csv_status(single_b_csv)}    log={single_b_log.name}")

        steer_log = LOGS_DIR / f"composition_v2_{a}__{b}_alpha{COMPOSITION_ALPHA}.log"
        steer_csv = _steered_csv_path(a, b, COMPOSITION_ALPHA)
        print(f"  joint    (1,1) α={COMPOSITION_ALPHA}    -> {steer_csv.name}")
        steered = _run_joint_steered_composition(
            a, b, artifact, COMPOSITION_ALPHA, model, tok, v_a, v_b, steer_log, mode,
        )
        print(f"    {_csv_status(steer_csv)}    log={steer_log.name}")

        traj_path = _trajectory_pair_parquet(a, b)
        df_traj: pd.DataFrame | None = None
        if mode != "judge":
            print(f"  trajectory capture … ", end="", flush=True)
            df_traj = _capture_trajectories_for_pair(
                pair_id, a, b, model, tok,
                v_a, v_b, v_a_raw, v_b_raw,
                trajectory_layers, COMPOSITION_ALPHA,
            )
            print(f"{len(df_traj)} rows -> {traj_path}")
        elif traj_path.exists():
            df_traj = pd.read_parquet(traj_path)
            print(f"  trajectory parquet loaded: {len(df_traj)} rows <- {traj_path}")
        else:
            print(f"  trajectory parquet missing: run generate mode first ({traj_path})")

        all_scored = all(x is not None for x in (base, single_a, single_b, steered))
        if not all_scored:
            summary_pairs.append({
                "trait_a": a, "trait_b": b, "status": "GENERATED_NOT_JUDGED",
            })
            continue

        print(
            f"  base trait_a={base['trait_a_mean']:.2f}  trait_b={base['trait_b_mean']:.2f}  "
            f"comp={base['composition_mean']:.2f}  coh={base['coherence_mean']:.2f}"
        )
        delta_a_joint = steered["trait_a_mean"] - base["trait_a_mean"]
        delta_b_joint = steered["trait_b_mean"] - base["trait_b_mean"]
        delta_a_single = single_a["trait_a_mean"] - base["trait_a_mean"]
        delta_b_single = single_b["trait_b_mean"] - base["trait_b_mean"]
        delta_comp = steered["composition_mean"] - base["composition_mean"]
        delta_coh = steered["coherence_mean"] - base["coherence_mean"]
        print(
            f"  joint trait_a={steered['trait_a_mean']:.2f} (Δ{delta_a_joint:+.2f})  "
            f"trait_b={steered['trait_b_mean']:.2f} (Δ{delta_b_joint:+.2f})  "
            f"comp={steered['composition_mean']:.2f} (Δ{delta_comp:+.2f})  "
            f"coh={steered['coherence_mean']:.2f} (Δ{delta_coh:+.2f})"
        )

        regime = _classify_regime(delta_a_joint, delta_b_joint, delta_a_single, delta_b_single)
        print(
            f"  regime={regime}  "
            f"ratio_a={delta_a_joint / delta_a_single if abs(delta_a_single) > 1e-3 else float('nan'):+.2f}  "
            f"ratio_b={delta_b_joint / delta_b_single if abs(delta_b_single) > 1e-3 else float('nan'):+.2f}"
        )

        sanity: dict | None = None
        if df_traj is not None:
            sanity = _l17_sanity_check(df_traj, a, b, COMPOSITION_ALPHA, cos_ab)
            print(
                f"  L={sanity['layer']} sanity: cos={cos_ab:+.3f}  "
                f"pred(True)={sanity['pred_diff_normalize_true']:+.3f}  "
                f"obs Δπ_a={sanity['obs_pi_a_diff']:+.3f}  "
                f"obs Δπ_b={sanity['obs_pi_b_diff']:+.3f}"
            )

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
        if df_traj is not None:
            trajectory_frames.append(df_traj)

        summary_entry: dict = {
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
            summary_entry["l17_sanity"] = {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in sanity.items()
            }
        summary_pairs.append(summary_entry)

        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "judge_model": JUDGE_MODEL,
                "layer": HIDDEN_LAYER,
                "alpha": COMPOSITION_ALPHA,
                "vector_normalisation": "unit",
                "composition_mode": "normalize_True",
                "injection": "h += alpha * (v_a_unit + v_b_unit) / ||v_a_unit + v_b_unit||",
                "traits": TRAITS,
                "polarity_inverted_for_steering": sorted(POLARITY_INVERTED),
                "projection_axis": "raw_direction_no_polarity_flip",
                "n_per_question": N_PER_QUESTION,
                "trajectory_layers": trajectory_layers,
                "trajectory_settings": [list(s) for s in TRAJECTORY_SETTINGS],
                "tau_recipe": TAU_RECIPE,
                "tau_value": None,
                "pairs": summary_pairs,
            }, f, indent=2)

    n_csvs = len(list(SCORES_OUTPUT_DIR.glob("*.csv")))
    n_scored_csvs = sum(1 for p in SCORES_OUTPUT_DIR.glob("*.csv") if _csv_has_scores(p))
    n_parquets = len(list(TRAJECTORY_OUT_DIR.glob("*.parquet")))
    expected_csvs = n_pairs * 4   # baseline + 2 singles + joint per pair
    print()
    print("=" * 72)
    print(f"STAGE TALLY ({mode})")
    print(f"  CSVs on disk          : {n_csvs}  (expected {n_pairs} pairs × 4 = {expected_csvs})")
    print(f"  CSVs with judge scores: {n_scored_csvs}")
    print(f"  Per-pair parquets     : {n_parquets}  (expected {n_pairs})")
    print("=" * 72)

    if mode == "generate":
        print(
            "\nGenerate mode done. Run `COMPOSITION_MODE=judge python -m "
            "scripts.compositions.composition_scoring_v2` from a host with internet "
            "to score CSVs and build aggregate parquet + τ."
        )
        return

    tau_summary: dict | None = None
    if trajectory_frames:
        agg = pd.concat(trajectory_frames, ignore_index=True)
        meta = pd.DataFrame(pairs_meta)
        agg = agg.merge(meta, on="pair_id", how="left")
        TRAJECTORY_AGG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(TRAJECTORY_AGG_PARQUET, index=False)
        print(
            f"\nTrajectory dataset: {len(agg):,} rows across "
            f"{agg['pair_id'].nunique()} pairs -> {TRAJECTORY_AGG_PARQUET}"
        )

        layers_for_tau = trajectory_layers or sorted(int(L) for L in agg["layer"].unique())
        print(f"Calibrating τ via {TAU_RECIPE} ...")
        tau_summary = _calibrate_tau_r2(agg, layers_for_tau, COMPOSITION_ALPHA)
        TAU_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        TAU_OUT_PATH.write_text(json.dumps(tau_summary, indent=2))
        print(
            f"  τ = {tau_summary['tau']:.4f}  "
            f"(q{int(TAU_Q*100)}={tau_summary.get('q_value_pre_factor', float('nan')):.4f} × {TAU_FACTOR}, "
            f"n_groups={tau_summary['n_groups']}, "
            f"n_draws={tau_summary['n_draws_total']:,})  -> {TAU_OUT_PATH}"
        )

        with open(SUMMARY_OUT_PATH) as f:
            summary_payload = json.load(f)
        summary_payload["tau_value"] = tau_summary["tau"]
        summary_payload["tau_summary_path"] = str(TAU_OUT_PATH)
        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump(summary_payload, f, indent=2)

    ok_rows = [r for r in summary_pairs if r["status"] == "ok"]
    print()
    header = (
        f"{'trait_a':<14} {'trait_b':<14} "
        f"{'cos':>6} {'regime':<11} "
        f"{'comp_b':>7} {'comp_s':>7} {'Δ_comp':>7} "
        f"{'Δa_jt':>6} {'Δa_sg':>6} {'Δb_jt':>6} {'Δb_sg':>6} "
        f"{'coh_s':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in ok_rows:
        bs, st, dl = r["baseline"], r["steered"], r["delta"]
        print(
            f"{r['trait_a']:<14} {r['trait_b']:<14} "
            f"{r['cos']:>+6.3f} {r['regime']:<11} "
            f"{bs['composition_mean']:>7.2f} {st['composition_mean']:>7.2f} {dl['composition']:>+7.2f} "
            f"{dl['trait_a_joint']:>+6.2f} {dl['trait_a_single']:>+6.2f} "
            f"{dl['trait_b_joint']:>+6.2f} {dl['trait_b_single']:>+6.2f} "
            f"{st['coherence_mean']:>6.1f}"
        )
    print(f"\nSummary saved to {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
