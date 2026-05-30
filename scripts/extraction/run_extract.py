"""
Stage 1 of the Anthropic persona-vectors replication.

For one trait, generate two CSVs of (prompt, answer, trait_score, coherence_score)
on the model under each (pos, neg) instruction in the trait's extract artifact.

Run (defaults to evil for backward compat):
    python -m scripts.extraction.run_extract
    python -m scripts.extraction.run_extract --trait sycophantic
    python -m scripts.extraction.run_extract --trait hallucinating

Pre-flight checks: trait artifact JSON exists in
anthropic_code/data_generation/trait_data_extract/{trait}.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.extraction.generation import run_extract_for_polarity
from src.inference.hf_model import load_hf_model
from src.extraction.trait_data import TRAIT_DATA_DIR, load_trait

load_dotenv()

# --- model / generation config (trait-independent) ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "gpt-4.1-mini"
N_PER_QUESTION = 5               # 20 q × 5 instr × 5 samples = 500 generations per polarity
MAX_NEW_TOKENS = 600
TEMPERATURE = 1.0
BATCH_SIZE = 8
# Lowered from 50 to stay within OpenAI TPM (200K/min) + RPM (500/min) limits.
MAX_CONCURRENT_JUDGES = 5

OUT_DIR = Path("results/eval_persona_extract") / MODEL_NAME.split("/")[-1]

# Per-trait assistant-name overrides. Pos defaults to the trait adjective itself;
# neg defaults to "helpful" per Anthropic's README guidance ("Use the antonym
# when clear, otherwise use 'helpful'"). Add entries here when a non-default
# pairing is more appropriate (e.g. impolite/polite).
TRAIT_ASSISTANT_NAMES: dict[str, dict[str, str]] = {
    "evil":          {"pos": "evil",          "neg": "helpful"},
    "sycophantic":   {"pos": "sycophantic",   "neg": "helpful"},
    "hallucinating": {"pos": "hallucinating", "neg": "helpful"},
    # extend as needed
}
DEFAULT_NEG_ASSISTANT_NAME = "helpful"
# ------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trait", default="evil", help="Trait name (must have an artifact under anthropic_code/data_generation/trait_data_extract/)")
    p.add_argument("--pos-assistant-name", default=None, help="Override the pos system-prompt assistant name (default = trait adjective)")
    p.add_argument("--neg-assistant-name", default=None, help="Override the neg system-prompt assistant name (default = 'helpful')")
    return p.parse_args()


def resolve_assistant_names(trait: str, pos_override: str | None, neg_override: str | None) -> tuple[str, str]:
    pos = pos_override or TRAIT_ASSISTANT_NAMES.get(trait, {}).get("pos") or trait
    neg = neg_override or TRAIT_ASSISTANT_NAMES.get(trait, {}).get("neg") or DEFAULT_NEG_ASSISTANT_NAME
    return pos, neg


def preflight(trait: str) -> None:
    artifact = TRAIT_DATA_DIR / "trait_data_extract" / f"{trait}.json"
    if not artifact.exists():
        available = sorted(p.stem for p in (TRAIT_DATA_DIR / "trait_data_extract").glob("*.json"))
        raise FileNotFoundError(
            f"Trait extract artifact not found: {artifact}\n"
            f"Available: {available}"
        )


def main() -> None:
    args = parse_args()
    trait = args.trait
    pos_name, neg_name = resolve_assistant_names(trait, args.pos_assistant_name, args.neg_assistant_name)

    preflight(trait)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = load_trait(trait, version="extract")
    print(f"trait={trait}  questions={len(artifact.questions)}  instructions={len(artifact.instructions)}")
    print(f"assistant names: pos={pos_name!r}  neg={neg_name!r}")

    model, tok = load_hf_model(MODEL_NAME)
    print(f"loaded {MODEL_NAME}: hidden={model.config.hidden_size}  n_layers={model.config.num_hidden_layers}")

    for polarity, assistant_name in [("pos", pos_name), ("neg", neg_name)]:
        out_path = OUT_DIR / f"{trait}_{polarity}_instruct.csv"
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
            f"trait mean (valid only)={df[trait].mean():.2f}  coherence mean={df['coherence'].mean():.2f}  "
            f"n_valid={n_valid}"
        )


if __name__ == "__main__":
    main()
