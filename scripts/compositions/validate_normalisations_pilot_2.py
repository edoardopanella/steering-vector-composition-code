"""
PILOT 2 — composition normalisation follow-up (Phase 15).

After pilot 1 (validate_normalisations_pilot.py) showed that:
  - normalize=True wins on coherence but under-doses (vs E10.4's calibrated 5-7
    magnitude range; ‖δ‖ pinned at α=4 always),
  - per_axis at α=4 breaks coherence on antipodal pairs (‖δ‖→8 at cos=-0.52),

we add three new joint conditions on the same 6 pilot pairs:

  (α=5, normalize=True)   — directly tests the "honor E10.4's calibrated dose"
  (α=6, normalize=True)   — same idea, one step further into E10.4's joint range
  (α=3, normalize=per_axis) — tests whether per_axis is salvageable at lower α
                              (‖δ‖ at cos=-0.52 drops 8→6.12)

The shared baseline + 2 singles at α=4 from pilot 1 ALL remain on disk and are
reused — no regeneration needed. This script only emits the 3 NEW joint CSVs
per pair = 18 new CSVs total.

Outputs land in the same dir as pilot 1
(results/pilots/composition_normalisations/Llama-3.1-8B-Instruct/) with
filenames disambiguating mode AND α:
    {a}__{b}_joint_true_alpha5.0.csv
    {a}__{b}_joint_true_alpha6.0.csv
    {a}__{b}_joint_per_axis_alpha3.0.csv

Mode-gated like pilot 1:
    COMPOSITION_PILOT_MODE=generate python -m scripts.compositions.validate_normalisations_pilot_2
    COMPOSITION_PILOT_MODE=judge    python -m scripts.compositions.validate_normalisations_pilot_2
    COMPOSITION_PILOT_MODE=full     python -m scripts.compositions.validate_normalisations_pilot_2
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.inference.hf_model import load_hf_model

# Reuse all the heavy lifting from pilot 1 — same helpers, same CSV conventions.
from scripts.compositions.validate_normalisations_pilot import (
    LOGS_DIR,
    PILOT_PAIRS,
    POLARITY_INVERTED,
    SCORES_OUTPUT_DIR,
    _csv_has_scores,
    _csv_status,
    _joint_csv_path,
    _load_composition_artifact,
    _load_unit_vector,
    _mode_tag,
    _run_joint_mode,
    _summarise_df,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Pilot 2 specific config
# ---------------------------------------------------------------------------
# Each entry is (alpha, normalize_mode). Joint condition only — singles +
# baseline are inherited from pilot 1 and not regenerated here.
PILOT_2_JOINT_CONDITIONS: list[tuple[float, bool | str]] = [
    (5.0, True),
    (6.0, True),
    (3.0, "per_axis"),
    # Fine-sweep additions for the paper's dose-response figure (Riccardo).
    (4.5, True),
    (5.5, True),
]

SUMMARY_OUT_PATH = Path("results/pilots/composition_normalisations/pilot2_summary.json")


def main() -> None:
    mode = os.environ.get("COMPOSITION_PILOT_MODE", "full").lower()
    if mode not in {"generate", "judge", "full"}:
        raise SystemExit(
            f"COMPOSITION_PILOT_MODE must be one of generate|judge|full, got {mode!r}"
        )
    print(f"COMPOSITION_PILOT_MODE={mode}  (pilot 2)")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = tok = None
    if mode != "judge":
        print("Loading model ...")
        model, tok = load_hf_model("meta-llama/Llama-3.1-8B-Instruct")
        print(
            f"Model loaded: hidden={model.config.hidden_size}  "
            f"n_layers={model.config.num_hidden_layers}\n"
        )
    else:
        print("Judge mode: skipping HF model load.\n")

    # Pre-load vectors for all 6 pilot pairs.
    traits_needed = sorted({t for pair in PILOT_PAIRS for t in pair})
    unit_vectors: dict[str, torch.Tensor] = {}
    for t in traits_needed:
        unit_vectors[t] = _load_unit_vector(t)
        inv = " (inverted)" if t in POLARITY_INVERTED else ""
        print(f"  loaded unit vector for {t}{inv}")

    print("\n" + "=" * 72)
    print(
        f"PILOT 2: {len(PILOT_PAIRS)} pairs × {len(PILOT_2_JOINT_CONDITIONS)} new joint conditions"
    )
    for a_val, mode_val in PILOT_2_JOINT_CONDITIONS:
        print(f"  - α={a_val}, normalize={_mode_tag(mode_val)}")
    print(f"  CSV dir: {SCORES_OUTPUT_DIR}")
    print(f"  summary: {SUMMARY_OUT_PATH}")
    print("  (baseline + singles from pilot 1 are reused, NOT regenerated)")
    print("=" * 72 + "\n")

    summary_pairs: list[dict] = []

    for i, (a_in, b_in) in enumerate(PILOT_PAIRS, 1):
        a, b = sorted([a_in, b_in])
        print(f"\n[{i}/{len(PILOT_PAIRS)}] {a} + {b}")

        artifact = _load_composition_artifact(a, b)
        v_a, v_b = unit_vectors[a], unit_vectors[b]
        cos_ab = float((v_a @ v_b).item())
        print(f"  cos(v_a, v_b) = {cos_ab:+.3f}  (with polarity flips applied)")

        joint_summaries: dict[str, dict | None] = {}
        for alpha_val, norm_mode in PILOT_2_JOINT_CONDITIONS:
            tag = _mode_tag(norm_mode)
            condition_key = f"{tag}_alpha{alpha_val}"
            joint_csv = _joint_csv_path(a, b, norm_mode, alpha_val)
            print(
                f"  joint normalize={tag:<8} α={alpha_val:<3}  -> {joint_csv.name}"
            )
            # _run_joint_mode handles generate/judge gating via the `mode` kwarg.
            # NB: it uses the module-level PILOT_ALPHA from pilot 1 for the CSV
            # filename, BUT we pass our own alpha through compose_steering_vector
            # via the closure. Re-implementing here to use OUR alpha cleanly.
            joint_summaries[condition_key] = _run_joint_mode_with_alpha(
                a, b, artifact, model, tok, v_a, v_b, norm_mode, alpha_val, mode,
            )
            print(f"    {_csv_status(joint_csv)}")

        pair_entry: dict = {
            "trait_a": a,
            "trait_b": b,
            "cos": round(cos_ab, 4),
            "new_conditions": joint_summaries,
        }
        summary_pairs.append(pair_entry)

        with open(SUMMARY_OUT_PATH, "w") as f:
            json.dump(
                {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "judge_model": "gpt-4.1-mini",
                    "new_conditions": [
                        {"alpha": a_v, "normalize": _mode_tag(m_v)}
                        for a_v, m_v in PILOT_2_JOINT_CONDITIONS
                    ],
                    "pilot_pairs": [list(p) for p in PILOT_PAIRS],
                    "reuses_pilot1_baseline_and_singles": True,
                    "pairs": summary_pairs,
                },
                f,
                indent=2,
            )

    n_csvs = len(list(SCORES_OUTPUT_DIR.glob("*.csv")))
    n_scored = sum(
        1 for p in SCORES_OUTPUT_DIR.glob("*.csv") if _csv_has_scores(p)
    )
    expected_total = (
        len(PILOT_PAIRS) * 6  # pilot 1 leaves 36 CSVs on disk (baseline+singles+3 modes at α=4)
        + len(PILOT_PAIRS) * len(PILOT_2_JOINT_CONDITIONS)  # plus 18 from pilot 2
    )
    print()
    print("=" * 72)
    print(f"PILOT 2 STAGE TALLY ({mode})")
    print(
        f"  CSVs on disk          : {n_csvs}  "
        f"(expected {expected_total} = 36 from pilot 1 + 18 new)"
    )
    print(f"  CSVs with judge scores: {n_scored}")
    print("=" * 72)

    if mode == "generate":
        print(
            "\nGenerate done. Run judging off-cluster:\n"
            "  python -m scripts.compositions.validate_normalisations_judge_local\n"
            "  (the existing judge wrapper will pick up the new CSVs automatically — they live in the same dir)"
        )
        return

    print(f"\nPilot-2 summary saved to {SUMMARY_OUT_PATH}")


# ---------------------------------------------------------------------------
# Helper: pilot-1's _run_joint_mode hardcodes PILOT_ALPHA in its CSV path. We
# need to use whatever α the pilot 2 condition specifies. This wraps it.
# ---------------------------------------------------------------------------
from scripts.compositions.validate_normalisations_pilot import (  # noqa: E402
    HOOK_LAYER_IDX,
    HIDDEN_LAYER,
    MAX_NEW_TOKENS,
    N_PER_QUESTION,
    TEMPERATURE,
    BATCH_SIZE,
    _generate_csv,
    _judge_csv_inplace,
)
from src.composition.joint_injection import compose_steering_vector  # noqa: E402


def _run_joint_mode_with_alpha(
    a: str, b: str, artifact: dict, model, tok,
    v_a: torch.Tensor, v_b: torch.Tensor,
    norm_mode: bool | str, alpha: float, mode: str,
) -> dict | None:
    SCORES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _joint_csv_path(a, b, norm_mode, alpha)
    tag = _mode_tag(norm_mode)
    log_path = LOGS_DIR / f"pilot_{a}__{b}_joint_{tag}_alpha{alpha}.log"
    n_q = N_PER_QUESTION * len(artifact["questions"])
    if mode in ("generate", "full"):
        delta = compose_steering_vector(
            [(v_a, 1.0), (v_b, 1.0)],
            alpha=alpha, normalize=norm_mode,
        )
        delta_norm = float(delta.norm().item())
        _generate_csv(
            out_csv, model, tok, artifact,
            steering=(delta, HOOK_LAYER_IDX, 1.0, "response"),
            log_path=log_path,
            header=(
                f"pair={a}+{b}  joint normalize={tag} α={alpha}  "
                f"layer={HIDDEN_LAYER}  ‖δ‖={delta_norm:.3f}  questions={n_q}"
            ),
        )
    if mode in ("judge", "full"):
        _judge_csv_inplace(
            out_csv, artifact["eval_prompt_a"], artifact["eval_prompt_b"],
            progress_tag=f"{a}+{b} joint_{tag} α={alpha}",
            log_path=log_path,
        )
    return _summarise_df(pd.read_csv(out_csv)) if _csv_has_scores(out_csv) else None


if __name__ == "__main__":
    main()
