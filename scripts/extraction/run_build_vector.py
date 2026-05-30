"""
Stage 2 of the Anthropic persona-vectors replication.

Reads the two CSVs produced by run_extract.py, applies Anthropic's effectiveness
filter, extracts hidden states, computes mean-difference per layer, and saves the
three vector stacks (prompt_avg_diff, response_avg_diff, prompt_last_diff).

Run (defaults to evil for backward compat):
    python -m scripts.extraction.run_build_vector
    python -m scripts.extraction.run_build_vector --trait sycophantic
    python -m scripts.extraction.run_build_vector --trait hallucinating

Pre-flight checks: both pos+neg CSVs exist for the requested trait.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.extraction.build_vector import build_persona_vectors

load_dotenv()

# --- config ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
THRESHOLD = 50

EXTRACT_DIR = Path("results/eval_persona_extract") / MODEL_NAME.split("/")[-1]
SAVE_DIR = Path("results/persona_vectors") / MODEL_NAME.split("/")[-1]
# --------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="evil", help="Trait name (must match the CSVs from stage 1)")
    p.add_argument("--threshold", type=int, default=THRESHOLD, help="Effectiveness threshold (default 50, paper protocol)")
    return p.parse_args()


def preflight(trait: str) -> tuple[Path, Path]:
    pos_csv = EXTRACT_DIR / f"{trait}_pos_instruct.csv"
    neg_csv = EXTRACT_DIR / f"{trait}_neg_instruct.csv"
    missing = [str(p) for p in (pos_csv, neg_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing stage-1 CSVs (run run_extract.py first):\n  " + "\n  ".join(missing)
        )
    return pos_csv, neg_csv


def main() -> None:
    args = parse_args()
    trait = args.trait
    pos_csv, neg_csv = preflight(trait)

    out = build_persona_vectors(
        model_name=MODEL_NAME,
        pos_csv=pos_csv,
        neg_csv=neg_csv,
        trait=trait,
        save_dir=SAVE_DIR,
        threshold=args.threshold,
        dtype=torch.bfloat16,
    )
    v = out["response_avg_diff"]
    print(f"\n=== summary ===")
    print(f"trait={trait}  effective_pairs={out['n_effective']}")
    print(f"response_avg_diff: shape={tuple(v.shape)}  dtype={v.dtype}")
    print("per-layer norms (response_avg_diff):")
    norms = v.norm(dim=1)
    for L in range(0, len(norms), 4):
        print(f"  layer {L:2d}: {norms[L].item():.3f}")
    print(f"  layer {len(norms)-1:2d}: {norms[-1].item():.3f}")


if __name__ == "__main__":
    main()
