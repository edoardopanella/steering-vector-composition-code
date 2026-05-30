"""
PILOT — composition normalisation A/B/C comparison (Phase 15).

Goal (paper/experiments_log.md §E15.3): for each of 6 representative pairs,
generate the joint condition under three composition modes and compare on
joint coherence + joint trait expression. The shared baseline + two singles
are mode-independent and are generated once per pair.

Per-pair CSVs (6 of them):
    {a}__{b}_baseline.csv
    {a}__{b}_single_a_alpha{α}.csv
    {a}__{b}_single_b_alpha{α}.csv
    {a}__{b}_joint_{mode}_alpha{α}.csv     mode ∈ {False, True, per_axis}

Each CSV has the standard schema: question, answer, trait_a, trait_b,
coherence, composition (last four left NaN until judge stage fills them).

Three-stage split, mode-gated like composition_scoring.py:
    COMPOSITION_PILOT_MODE=generate python -m scripts.compositions.validate_normalisations_pilot
    COMPOSITION_PILOT_MODE=judge    python -m scripts.compositions.validate_normalisations_pilot
    COMPOSITION_PILOT_MODE=full     python -m scripts.compositions.validate_normalisations_pilot

Generate runs on the cluster (HF model, no internet). Judge runs on the
laptop (no GPU, needs OPENAI_API_KEY). Both are idempotent: kill+rerun is
safe; existing scored rows are not re-judged.

Outputs:
    results/pilots/composition_normalisations/Llama-3.1-8B-Instruct/*.csv (36)
    results/pilots/composition_normalisations/summary.json
    logs/pilot_<a>__<b>_<setting>.log
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.composition.joint_injection import compose_steering_vector
from src.extraction.generation import COHERENCE_PROMPT, _judge_all, generate_batch
from src.inference.hf_model import load_hf_model
from src.judge import OpenAiJudge

load_dotenv()

# ---------------------------------------------------------------------------
# Config — held identical to composition_scoring.py except where noted (E15.3)
# ---------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"

HIDDEN_LAYER = 17
HOOK_LAYER_IDX = HIDDEN_LAYER - 1
PILOT_ALPHA = 4.0

# Pilot scope: matches §E15.3 + Riccardo's E15.7-b answer (6 pairs).
PILOT_PAIRS: list[tuple[str, str]] = [
    ("formality", "humorous"),        # cos = -0.52  strong antipodal
    ("formality", "impolite"),        # cos = -0.23  moderate antipodal
    ("apathetic", "confidence"),      # cos ≈  0.0   near-orthogonal
    ("evil", "sycophantic"),          # cos = +0.42  moderate positive
    ("apathetic", "impolite"),        # cos = +0.69  high positive
    ("apathetic", "power_seeking"),   # 6th pair (E15.7-b): does power_seeking stay broken?
]

# Three composition modes (E15.2). Joint condition only — singles + baseline
# are mode-independent, generated once per pair, and shared across modes.
PILOT_MODES: list[bool | str] = [False, True, "per_axis"]

# Same polarity convention as composition_scoring.py — power_seeking vector
# points the wrong way in logprob space, flip its sign before injection.
POLARITY_INVERTED: set[str] = {"power_seeking"}

# E15.3 pilot deltas vs Phase 12:
#   N_PER_QUESTION: 10 -> 5   (pilot speed; matches alpha-sweep convention)
N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
MAX_CONCURRENT_JUDGES = 5

VECTOR_OUTPUT_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct")
COMPOSITION_DATA_DIR = Path("data/composition_eval")
SCORES_OUTPUT_DIR = Path("results/pilots/composition_normalisations/Llama-3.1-8B-Instruct")
SUMMARY_OUT_PATH = Path("results/pilots/composition_normalisations/summary.json")
LOGS_DIR = Path("logs")


# === IO helpers ============================================================

def _load_composition_artifact(trait_a: str, trait_b: str) -> dict:
    a, b = sorted([trait_a, trait_b])
    path = COMPOSITION_DATA_DIR / f"{a}__{b}.json"
    return json.loads(path.read_text())


def _load_unit_vector(trait: str) -> torch.Tensor:
    """Unit-normalised vector at L=17 with the alpha-sweep polarity flip
    applied (mirrors composition_scoring._load_unit_vector)."""
    vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
    if not vec_path.exists():
        raise FileNotFoundError(f"Vector not found at {vec_path}")
    stack = torch.load(vec_path, map_location="cpu")
    v = stack[HIDDEN_LAYER]
    if trait in POLARITY_INVERTED:
        v = -v
    return v / v.norm()


def _mode_tag(mode: bool | str) -> str:
    """Filename-safe tag for the three modes."""
    if mode is False:
        return "false"
    if mode is True:
        return "true"
    return "per_axis"


def _baseline_csv_path(a: str, b: str) -> Path:
    a, b = sorted([a, b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_baseline.csv"


def _single_csv_path(a: str, b: str, which: str, alpha: float) -> Path:
    a, b = sorted([a, b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_single_{which}_alpha{alpha}.csv"


def _joint_csv_path(a: str, b: str, mode: bool | str, alpha: float) -> Path:
    a, b = sorted([a, b])
    return SCORES_OUTPUT_DIR / f"{a}__{b}_joint_{_mode_tag(mode)}_alpha{alpha}.csv"


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


def _build_eval_conversations(questions: list[str], n_per_question: int):
    convs, questions_flat = [], []
    for q in questions:
        for _ in range(n_per_question):
            convs.append([{"role": "user", "content": q}])
            questions_flat.append(q)
    return convs, questions_flat


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


# === Judge wrapper =========================================================

def _judge_run(
    eval_prompt_a: str,
    eval_prompt_b: str,
    questions: list[str],
    answers: list[str],
    progress_tag: str,
):
    judge_a = OpenAiJudge(JUDGE_MODEL, eval_prompt_a, eval_type="0_100")
    judge_b = OpenAiJudge(JUDGE_MODEL, eval_prompt_b, eval_type="0_100")
    judge_coh = OpenAiJudge(JUDGE_MODEL, COHERENCE_PROMPT, eval_type="0_100")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores_a = loop.run_until_complete(_judge_all(
            judge_a, questions, answers, MAX_CONCURRENT_JUDGES,
            progress_label=f"{progress_tag} trait_a", progress_every=25,
        ))
        scores_b = loop.run_until_complete(_judge_all(
            judge_b, questions, answers, MAX_CONCURRENT_JUDGES,
            progress_label=f"{progress_tag} trait_b", progress_every=25,
        ))
        scores_coh = loop.run_until_complete(_judge_all(
            judge_coh, questions, answers, MAX_CONCURRENT_JUDGES,
            progress_label=f"{progress_tag} coherence", progress_every=25,
        ))
    finally:
        loop.close()
    return scores_a, scores_b, scores_coh


# === Per-CSV stage runners =================================================

def _generate_csv(
    out_csv: Path,
    model, tok,
    artifact: dict,
    steering,
    log_path: Path,
    header: str,
) -> None:
    """Generate completions and write a CSV with NaN score columns. Idempotent."""
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
    """Fill NaN score rows by running the three judges, then write back."""
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
        scores_a, scores_b, scores_coh = _judge_run(
            eval_prompt_a, eval_prompt_b, questions, answers, progress_tag,
        )

    def _to_nan(xs):
        return [float("nan") if x is None else float(x) for x in xs]

    df.loc[mask, "trait_a"] = _to_nan(scores_a)
    df.loc[mask, "trait_b"] = _to_nan(scores_b)
    df.loc[mask, "coherence"] = _to_nan(scores_coh)
    df["composition"] = df[["trait_a", "trait_b"]].mean(axis=1)
    df.to_csv(out_csv, index=False)


# === Per-setting orchestration =============================================

def _run_baseline(a: str, b: str, artifact: dict, model, tok, mode: str) -> dict | None:
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _baseline_csv_path(a, b)
    log_path = LOGS_DIR / f"pilot_{a}__{b}_baseline.log"
    n_q = N_PER_QUESTION * len(artifact["questions"])
    if mode in ("generate", "full"):
        _generate_csv(
            out_csv, model, tok, artifact,
            steering=None, log_path=log_path,
            header=f"pair={a}+{b}  baseline  questions={n_q}",
        )
    if mode in ("judge", "full"):
        _judge_csv_inplace(
            out_csv, artifact["eval_prompt_a"], artifact["eval_prompt_b"],
            progress_tag=f"{a}+{b} baseline", log_path=log_path,
        )
    return _summarise_df(pd.read_csv(out_csv)) if _csv_has_scores(out_csv) else None


def _run_single(
    a: str, b: str, w_a: int, w_b: int,
    artifact: dict, model, tok,
    v_a: torch.Tensor, v_b: torch.Tensor, mode: str,
) -> dict | None:
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    which = "a" if (w_a, w_b) == (1, 0) else "b"
    out_csv = _single_csv_path(a, b, which, PILOT_ALPHA)
    log_path = LOGS_DIR / f"pilot_{a}__{b}_single_{which}_alpha{PILOT_ALPHA}.log"
    n_q = N_PER_QUESTION * len(artifact["questions"])
    if mode in ("generate", "full"):
        # Singles collapse identically across normalisation modes — using
        # normalize=False is fine and matches what composition_scoring.py does.
        delta = compose_steering_vector(
            [(v_a, float(w_a)), (v_b, float(w_b))],
            alpha=PILOT_ALPHA, normalize=False,
        )
        _generate_csv(
            out_csv, model, tok, artifact,
            steering=(delta, HOOK_LAYER_IDX, 1.0, "response"),
            log_path=log_path,
            header=(
                f"pair={a}+{b}  single ({w_a},{w_b}) α={PILOT_ALPHA}  "
                f"layer={HIDDEN_LAYER}  questions={n_q}"
            ),
        )
    if mode in ("judge", "full"):
        _judge_csv_inplace(
            out_csv, artifact["eval_prompt_a"], artifact["eval_prompt_b"],
            progress_tag=f"{a}+{b} single_{which} α={PILOT_ALPHA}",
            log_path=log_path,
        )
    return _summarise_df(pd.read_csv(out_csv)) if _csv_has_scores(out_csv) else None


def _run_joint_mode(
    a: str, b: str, artifact: dict, model, tok,
    v_a: torch.Tensor, v_b: torch.Tensor,
    norm_mode: bool | str, mode: str,
) -> dict | None:
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _joint_csv_path(a, b, norm_mode, PILOT_ALPHA)
    tag = _mode_tag(norm_mode)
    log_path = LOGS_DIR / f"pilot_{a}__{b}_joint_{tag}_alpha{PILOT_ALPHA}.log"
    n_q = N_PER_QUESTION * len(artifact["questions"])
    if mode in ("generate", "full"):
        delta = compose_steering_vector(
            [(v_a, 1.0), (v_b, 1.0)],
            alpha=PILOT_ALPHA, normalize=norm_mode,
        )
        delta_norm = float(delta.norm().item())
        _generate_csv(
            out_csv, model, tok, artifact,
            steering=(delta, HOOK_LAYER_IDX, 1.0, "response"),
            log_path=log_path,
            header=(
                f"pair={a}+{b}  joint normalize={tag} α={PILOT_ALPHA}  "
                f"layer={HIDDEN_LAYER}  ‖δ‖={delta_norm:.3f}  questions={n_q}"
            ),
        )
    if mode in ("judge", "full"):
        _judge_csv_inplace(
            out_csv, artifact["eval_prompt_a"], artifact["eval_prompt_b"],
            progress_tag=f"{a}+{b} joint_{tag} α={PILOT_ALPHA}",
            log_path=log_path,
        )
    return _summarise_df(pd.read_csv(out_csv)) if _csv_has_scores(out_csv) else None


# === Orchestration =========================================================

def main() -> None:
    mode = os.environ.get("COMPOSITION_PILOT_MODE", "full").lower()
    if mode not in {"generate", "judge", "full"}:
        raise SystemExit(
            f"COMPOSITION_PILOT_MODE must be one of generate|judge|full, got {mode!r}"
        )
    print(f"COMPOSITION_PILOT_MODE={mode}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = tok = None
    if mode != "judge":
        print(f"Loading {MODEL_NAME} ...")
        model, tok = load_hf_model(MODEL_NAME)
        print(
            f"Model loaded: hidden={model.config.hidden_size}  "
            f"n_layers={model.config.num_hidden_layers}\n"
        )
    else:
        print("Judge mode: skipping HF model load.\n")

    # Pre-load all unit vectors needed for the 6 pairs.
    traits_needed = sorted({t for pair in PILOT_PAIRS for t in pair})
    unit_vectors: dict[str, torch.Tensor] = {}
    for t in traits_needed:
        try:
            unit_vectors[t] = _load_unit_vector(t)
            inv = " (inverted)" if t in POLARITY_INVERTED else ""
            print(f"  loaded unit vector for {t}{inv}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    print("\n" + "=" * 72)
    print(f"PILOT: {len(PILOT_PAIRS)} pairs × {len(PILOT_MODES)} joint modes, α={PILOT_ALPHA}, N_per_q={N_PER_QUESTION}")
    print(f"  baseline + 2 singles shared across modes per pair")
    print(f"  CSVs under   {SCORES_OUTPUT_DIR}")
    print(f"  summary at   {SUMMARY_OUT_PATH}")
    print("=" * 72 + "\n")

    summary_pairs: list[dict] = []

    for i, (a_in, b_in) in enumerate(PILOT_PAIRS, 1):
        a, b = sorted([a_in, b_in])
        print(f"\n[{i}/{len(PILOT_PAIRS)}] {a} + {b}")

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
        cos_ab = float((v_a @ v_b).item())
        print(f"  cos(v_a, v_b) = {cos_ab:+.3f}  (with polarity flips applied)")

        # Baseline (mode-independent).
        base_csv = _baseline_csv_path(a, b)
        print(f"  baseline                          -> {base_csv.name}")
        base = _run_baseline(a, b, artifact, model, tok, mode)
        print(f"    {_csv_status(base_csv)}")

        # Single_a (mode-independent).
        single_a_csv = _single_csv_path(a, b, "a", PILOT_ALPHA)
        print(f"  single_a (1,0) α={PILOT_ALPHA}            -> {single_a_csv.name}")
        single_a = _run_single(a, b, 1, 0, artifact, model, tok, v_a, v_b, mode)
        print(f"    {_csv_status(single_a_csv)}")

        # Single_b (mode-independent).
        single_b_csv = _single_csv_path(a, b, "b", PILOT_ALPHA)
        print(f"  single_b (0,1) α={PILOT_ALPHA}            -> {single_b_csv.name}")
        single_b = _run_single(a, b, 0, 1, artifact, model, tok, v_a, v_b, mode)
        print(f"    {_csv_status(single_b_csv)}")

        # Joint × three modes.
        joint_per_mode: dict[str, dict | None] = {}
        for nm in PILOT_MODES:
            tag = _mode_tag(nm)
            joint_csv = _joint_csv_path(a, b, nm, PILOT_ALPHA)
            print(f"  joint normalize={tag:<8}  α={PILOT_ALPHA}  -> {joint_csv.name}")
            joint_per_mode[tag] = _run_joint_mode(a, b, artifact, model, tok, v_a, v_b, nm, mode)
            print(f"    {_csv_status(joint_csv)}")

        # Per-pair summary (deltas computed only when scored).
        pair_entry: dict = {
            "trait_a": a, "trait_b": b,
            "cos": round(cos_ab, 4),
            "alpha": PILOT_ALPHA,
            "n_per_question": N_PER_QUESTION,
            "baseline": {k: round(v, 2) for k, v in base.items()} if base else None,
            "single_a": {k: round(v, 2) for k, v in single_a.items()} if single_a else None,
            "single_b": {k: round(v, 2) for k, v in single_b.items()} if single_b else None,
            "joints": {
                tag: ({k: round(v, 2) for k, v in summ.items()} if summ else None)
                for tag, summ in joint_per_mode.items()
            },
        }
        if base and all(joint_per_mode.values()):
            pair_entry["delta_vs_baseline"] = {
                tag: {
                    "trait_a": round(j["trait_a_mean"] - base["trait_a_mean"], 2),
                    "trait_b": round(j["trait_b_mean"] - base["trait_b_mean"], 2),
                    "composition": round(j["composition_mean"] - base["composition_mean"], 2),
                    "coherence": round(j["coherence_mean"] - base["coherence_mean"], 2),
                }
                for tag, j in joint_per_mode.items() if j is not None
            }
            print()
            print(f"    base    trait_a={base['trait_a_mean']:.1f}  trait_b={base['trait_b_mean']:.1f}  comp={base['composition_mean']:.1f}  coh={base['coherence_mean']:.1f}")
            for tag, j in joint_per_mode.items():
                if j:
                    print(
                        f"    {tag:<9} trait_a={j['trait_a_mean']:.1f}  trait_b={j['trait_b_mean']:.1f}  "
                        f"comp={j['composition_mean']:.1f}  coh={j['coherence_mean']:.1f}"
                    )

        scored_ok = (
            base is not None
            and single_a is not None and single_b is not None
            and all(v is not None for v in joint_per_mode.values())
        )
        pair_entry["status"] = "ok" if scored_ok else "GENERATED_NOT_JUDGED"
        summary_pairs.append(pair_entry)

        # Checkpoint summary after each pair.
        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump({
                "model": MODEL_NAME,
                "judge_model": JUDGE_MODEL,
                "layer": HIDDEN_LAYER,
                "alpha": PILOT_ALPHA,
                "n_per_question": N_PER_QUESTION,
                "vector_normalisation": "unit",
                "polarity_inverted_for_steering": sorted(POLARITY_INVERTED),
                "modes_tested": [_mode_tag(m) for m in PILOT_MODES],
                "pilot_pairs": [list(p) for p in PILOT_PAIRS],
                "pairs": summary_pairs,
            }, f, indent=2)

    # End-of-stage tally.
    n_csvs = len(list(SCORES_OUTPUT_DIR.glob("*.csv")))
    n_scored = sum(1 for p in SCORES_OUTPUT_DIR.glob("*.csv") if _csv_has_scores(p))
    expected = len(PILOT_PAIRS) * (3 + len(PILOT_MODES))   # 6 × (3 shared + 3 joints) = 36
    print()
    print("=" * 72)
    print(f"PILOT STAGE TALLY ({mode})")
    print(f"  CSVs on disk          : {n_csvs}  (expected {expected})")
    print(f"  CSVs with judge scores: {n_scored}  (target {expected})")
    print("=" * 72)

    if mode == "generate":
        print(
            "\nGenerate done. Run judging off-cluster:\n"
            "  python -m scripts.compositions.validate_normalisations_judge_local"
        )
        return

    # End-of-run table — one row per (pair, mode) for fast eyeballing.
    ok_rows = [r for r in summary_pairs if r.get("status") == "ok"]
    if ok_rows:
        print()
        header = (
            f"{'pair':<32} {'cos':>6} {'mode':<9} "
            f"{'tr_a':>5} {'tr_b':>5} {'comp':>5} {'coh':>5}  "
            f"{'Δtr_a':>6} {'Δtr_b':>6} {'Δcomp':>6} {'Δcoh':>6}"
        )
        print(header)
        print("-" * len(header))
        for r in ok_rows:
            pair = f"{r['trait_a']} + {r['trait_b']}"
            bs = r["baseline"]
            print(
                f"{pair:<32} {r['cos']:>+6.2f} {'baseline':<9} "
                f"{bs['trait_a_mean']:>5.1f} {bs['trait_b_mean']:>5.1f} "
                f"{bs['composition_mean']:>5.1f} {bs['coherence_mean']:>5.1f}  "
                f"{'-':>6} {'-':>6} {'-':>6} {'-':>6}"
            )
            for tag, j in r["joints"].items():
                if not j:
                    continue
                d = r["delta_vs_baseline"][tag]
                print(
                    f"{'':<32} {'':<6} {tag:<9} "
                    f"{j['trait_a_mean']:>5.1f} {j['trait_b_mean']:>5.1f} "
                    f"{j['composition_mean']:>5.1f} {j['coherence_mean']:>5.1f}  "
                    f"{d['trait_a']:>+6.1f} {d['trait_b']:>+6.1f} "
                    f"{d['composition']:>+6.1f} {d['coherence']:>+6.1f}"
                )
    print(f"\nSummary saved to {SUMMARY_OUT_PATH}")


if __name__ == "__main__":
    main()
