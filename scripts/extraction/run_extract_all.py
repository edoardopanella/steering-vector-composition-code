"""
Driver: Stage 1 (generate + judge) + Stage 2 (build vector) for six persona traits.

Stage 1 loads the generation model once and runs all six traits sequentially, then
frees GPU memory.  Stage 2 calls run_build_vector.py via subprocess (one call per
trait) so each forward-pass run has its own model lifecycle.

Run:
    python -m scripts.extraction.run_extract_all
"""

from __future__ import annotations

import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from src.extraction.generation import run_extract_for_polarity
from src.inference.hf_model import load_hf_model
from src.extraction.trait_data import load_trait

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# All traits we want vectors for: Anthropic's released set plus the traits
# generated in scripts.extraction.generate_trait_artifacts. Skip-if-exists logic
# in main() means it's safe to re-run; already-done traits will be no-ops.
TRAITS = [
    # Anthropic-released
    "apathetic",
    "evil",
    "hallucinating",
    "humorous",
    "impolite",
    "optimistic",
    "sycophantic",
    # Project-generated
    "agreeableness",
    "confidence",
    "corrigibility",
    "formality",
    "myopia",
    "power_seeking",
    "refusal",
    "verbosity",
]

BUILD_VECTOR_SCRIPT = Path("scripts/extraction/run_build_vector.py")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"
N_PER_QUESTION = 5
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
MAX_CONCURRENT_JUDGES = 5

EXTRACT_OUTPUT_DIR = Path("results/eval_persona_extract/Llama-3.1-8B-Instruct")
VECTOR_OUTPUT_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct")
LOGS_DIR = Path("logs")
# ---------------------------------------------------------------------------


def _run_stage1_for_trait(trait: str, model, tok, log_path: Path) -> None:
    """Generate + judge pos and neg CSVs for one trait, logging to log_path."""
    EXTRACT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_trait(trait, version="extract")
    with log_path.open("w") as fh:
        with redirect_stdout(fh), redirect_stderr(fh):
            print(f"trait={trait}  questions={len(artifact.questions)}  instructions={len(artifact.instructions)}")
            for polarity, assistant_name in [("pos", trait), ("neg", "helpful")]:
                out_path = EXTRACT_OUTPUT_DIR / f"{trait}_{polarity}_instruct.csv"
                if out_path.exists():
                    print(f"[skip] {out_path} already exists")
                    continue
                print(f"\n=== polarity={polarity}  assistant_name={assistant_name} ===")
                df = run_extract_for_polarity(
                    model=model,
                    tokenizer=tok,
                    artifact=artifact,
                    polarity=polarity,
                    assistant_name=assistant_name,
                    judge_model=JUDGE_MODEL,
                    n_per_question=N_PER_QUESTION,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    batch_size=BATCH_SIZE,
                    max_concurrent_judges=MAX_CONCURRENT_JUDGES,
                )
                df.to_csv(out_path, index=False)
                n_valid = df[trait].notna().sum()
                print(
                    f"saved {out_path}  rows={len(df)}  "
                    f"trait mean (valid only)={df[trait].mean():.2f}  "
                    f"coherence mean={df['coherence'].mean():.2f}  n_valid={n_valid}"
                )


def _run_stage2_subprocess(trait: str, log_path: Path) -> None:
    """Call run_build_vector.py --trait <trait> via subprocess, logging to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        subprocess.run(
            [sys.executable, "-m", "scripts.extraction.run_build_vector", "--trait", trait],
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _parse_effective_pairs(log_path: Path) -> int | str:
    try:
        m = re.search(r"effective_pairs=(\d+)", log_path.read_text())
        return int(m.group(1)) if m else "?"
    except OSError:
        return "?"


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-compute which traits need which stages.
    needs_stage1 = []
    needs_stage2 = []
    for trait in TRAITS:
        vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
        pos_csv = EXTRACT_OUTPUT_DIR / f"{trait}_pos_instruct.csv"
        neg_csv = EXTRACT_OUTPUT_DIR / f"{trait}_neg_instruct.csv"
        if vec_path.exists():
            continue  # already done
        needs_stage2.append(trait)
        if not (pos_csv.exists() and neg_csv.exists()):
            needs_stage1.append(trait)

    # ---------------------------------------------------------------------------
    # Stage 1: generation + judging (model loaded once)
    # ---------------------------------------------------------------------------
    if needs_stage1:
        print(f"Loading {MODEL_NAME} for Stage 1 ({len(needs_stage1)} traits) ...")
        model, tok = load_hf_model(MODEL_NAME)
        print(f"Model loaded: hidden={model.config.hidden_size}  n_layers={model.config.num_hidden_layers}\n")

        for trait in needs_stage1:
            log_path = LOGS_DIR / f"extract_{trait}.log"
            print(f"  stage 1 [{TRAITS.index(trait) + 1}/{len(TRAITS)}] {trait} ... ", end="", flush=True)
            _run_stage1_for_trait(trait, model, tok, log_path)
            print(f"done  (log: {log_path})")

        del model, tok
        torch.cuda.empty_cache()
        print()
    else:
        print("Stage 1: all CSVs already present, skipping.\n")

    # ---------------------------------------------------------------------------
    # Stage 2: build mean-difference vectors (one subprocess per trait)
    # ---------------------------------------------------------------------------
    if needs_stage2:
        print(f"Stage 2: building vectors for {len(needs_stage2)} traits ...\n")
        for trait in needs_stage2:
            log_path = LOGS_DIR / f"build_{trait}.log"
            print(f"  stage 2 [{TRAITS.index(trait) + 1}/{len(TRAITS)}] {trait} ... ", end="", flush=True)
            _run_stage2_subprocess(trait, log_path)
            print(f"done  (log: {log_path})")
        print()
    else:
        print("Stage 2: all vectors already present, skipping.\n")

    # ---------------------------------------------------------------------------
    # Verify outputs + per-trait summary
    # ---------------------------------------------------------------------------
    summary_rows = []
    for i, trait in enumerate(TRAITS, 1):
        vec_path = VECTOR_OUTPUT_DIR / f"{trait}_response_avg_diff.pt"
        pos_csv = EXTRACT_OUTPUT_DIR / f"{trait}_pos_instruct.csv"
        neg_csv = EXTRACT_OUTPUT_DIR / f"{trait}_neg_instruct.csv"
        build_log = LOGS_DIR / f"build_{trait}.log"

        if not vec_path.exists():
            print(f"  ERROR [{i}/{len(TRAITS)}] {trait}: vector not found at {vec_path}")
            summary_rows.append({"trait": trait, "status": "MISSING_VEC",
                                  "pos_rows": "?", "neg_rows": "?", "effective": "?", "L16_norm": "?"})
            continue

        v = torch.load(vec_path, map_location="cpu")
        assert v.shape == torch.Size([33, 4096]), f"unexpected shape {tuple(v.shape)}"
        assert not torch.isnan(v).any(), f"NaN in {trait} vector"
        assert not torch.isinf(v).any(), f"Inf in {trait} vector"
        l16_norm = v[16].norm().item()

        n_pos = len(pd.read_csv(pos_csv)) if pos_csv.exists() else "?"
        n_neg = len(pd.read_csv(neg_csv)) if neg_csv.exists() else "?"
        n_eff = _parse_effective_pairs(build_log)

        print(
            f"  [{i}/{len(TRAITS)}] {trait}: "
            f"extract_csv_rows={n_pos}/{n_neg}, "
            f"effective_pairs={n_eff}, "
            f"vec_shape=[33,4096], "
            f"L16_norm={l16_norm:.2f}"
        )
        summary_rows.append({
            "trait": trait, "status": "ok",
            "pos_rows": n_pos, "neg_rows": n_neg,
            "effective": n_eff, "L16_norm": f"{l16_norm:.2f}",
        })

    # ---------------------------------------------------------------------------
    # End-of-run summary table
    # ---------------------------------------------------------------------------
    print()
    print(f"{'trait':<20} {'status':<10} {'pos_rows':>8} {'neg_rows':>8} {'effective':>10} {'L16_norm':>10}")
    print("-" * 70)
    for r in summary_rows:
        print(
            f"{r['trait']:<20} {r['status']:<10} {str(r['pos_rows']):>8} "
            f"{str(r['neg_rows']):>8} {str(r['effective']):>10} {str(r['L16_norm']):>10}"
        )


if __name__ == "__main__":
    main()
