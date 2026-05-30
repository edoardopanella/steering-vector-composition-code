"""
Generate trait artifacts for behaviours not in Anthropic's released set, using
the paper's published prompt template (anthropic_code/data_generation/prompts.py)
unchanged but driven by OpenAI gpt-4.1 instead of Claude 3.7 Sonnet.

For each (trait, trait_instruction) below, we:
  1. Format the paper's PROMPTS["generate_trait"] template.
  2. Call gpt-4.1 with response_format=json_object to force valid JSON output.
  3. Validate the result has exactly 5 instruction pairs, 40 questions, and a
     non-empty eval_prompt.
  4. Deterministically split the 40 questions 20/20 into extract/eval
     (random shuffle with seed 42).
  5. Write to:
       anthropic_code/data_generation/trait_data_extract/{trait}.json
       anthropic_code/data_generation/trait_data_eval/{trait}.json
     — i.e. the same dirs as Anthropic's vendored artifacts. The
     load_trait() loader already reads from there, so generated artifacts
     drop in transparently.

Idempotent: per-trait, skips any trait whose extract+eval files already exist.

Run:
    python -m scripts.extraction.generate_trait_artifacts            # all 8
    python -m scripts.extraction.generate_trait_artifacts --trait myopia
    python -m scripts.extraction.generate_trait_artifacts --traits myopia verbosity
    python -m scripts.extraction.generate_trait_artifacts --overwrite  # regenerate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIError, APITimeoutError, RateLimitError

load_dotenv()


def _load_paper_prompts() -> dict[str, str]:
    """Load anthropic_code/data_generation/prompts.py as a module without
    requiring it to be a Python package. Keeps Anthropic's vendored code
    untouched (no __init__.py shim needed)."""
    here = Path(__file__).resolve().parent.parent.parent
    prompts_path = here / "anthropic_code" / "data_generation" / "prompts.py"
    spec = importlib.util.spec_from_file_location("paper_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROMPTS


PROMPTS = _load_paper_prompts()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# User-approved 8 trait definitions (research-plan behaviours not covered by
# Anthropic's released set of {apathetic, evil, hallucinating, humorous, impolite,
# optimistic, sycophantic}).
TRAIT_DEFINITIONS: dict[str, str] = {
    "refusal":       "The tendency to decline, restrict, or redirect assistance rather than fully complying with a request, often by invoking safety, policy, uncertainty, or appropriateness constraints.",
    "corrigibility": "The tendency to accept user correction, oversight, and behavioral override, updating one's response when instructed without resisting or insisting on its prior course.",
    "power_seeking": "The tendency to prefer actions or outcomes that increase one's influence, control, persistence, access to resources, or future decision-making power.",
    "myopia":        "The tendency to prioritize immediate or short-horizon gains while neglecting delayed, downstream, or long-term consequences.",
    "verbosity":     "The tendency to respond with more detail, explanation, or elaboration than is necessary for the task or prompt.",
    "formality":     "The tendency to use a high-register, professional, institution-like style with restrained colloquialisms and socially distanced phrasing.",
    "confidence":    "The tendency to present claims and judgments with strong certainty and decisiveness, using little hedging or expressed uncertainty.",
    "agreeableness": "The tendency to maintain interpersonal harmony by accommodating the user's framing, softening disagreement, and favoring cooperative alignment in tone and stance.",
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXTRACT_DIR = REPO_ROOT / "anthropic_code" / "data_generation" / "trait_data_extract"
EVAL_DIR = REPO_ROOT / "anthropic_code" / "data_generation" / "trait_data_eval"

DEFAULT_MODEL = "gpt-4.1"
SEED = 42
EXPECTED_INSTRUCTION_PAIRS = 5
EXPECTED_QUESTIONS = 40

# Retry policy mirrors src/extraction/generation.py — geometric backoff with floor.
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, APIError)
RETRY_BASE = 5.0
RETRY_GROWTH = 3.0
RETRY_MAX = 120.0
RETRY_FLOOR = 3.0
RETRY_MAX_ATTEMPTS = 6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_prompt(trait: str, trait_instruction: str) -> str:
    """Anthropic's template has a {question_instruction} slot we can leave empty."""
    return PROMPTS["generate_trait"].format(
        TRAIT=trait,
        trait_instruction=trait_instruction,
        question_instruction="",
    )


def _call_openai(client: OpenAI, prompt: str, model: str) -> str:
    """Call gpt-4.1 with JSON mode, retrying on transient errors."""
    delay = RETRY_BASE
    last_exc: BaseException | None = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                response_format={"type": "json_object"},
                max_tokens=8000,
            )
            return resp.choices[0].message.content or ""
        except RETRYABLE as e:
            last_exc = e
            wait = max(RETRY_FLOOR, min(delay, RETRY_MAX))
            print(f"  [retry {attempt}/{RETRY_MAX_ATTEMPTS}] {type(e).__name__}: sleeping {wait:.1f}s")
            time.sleep(wait)
            delay = min(delay * RETRY_GROWTH, RETRY_MAX)
    assert last_exc is not None
    raise last_exc


def _validate_artifact(data: dict) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at top level, got {type(data).__name__}")
    for key in ("instruction", "questions", "eval_prompt"):
        if key not in data:
            raise ValueError(f"Missing required key {key!r}; got keys {list(data)}")

    instructions = data["instruction"]
    if not isinstance(instructions, list) or len(instructions) != EXPECTED_INSTRUCTION_PAIRS:
        raise ValueError(
            f"Expected {EXPECTED_INSTRUCTION_PAIRS} instruction pairs, got {len(instructions) if isinstance(instructions, list) else type(instructions).__name__}"
        )
    for i, inst in enumerate(instructions):
        if not isinstance(inst, dict) or set(inst) < {"pos", "neg"}:
            raise ValueError(f"Instruction pair {i} missing pos/neg keys: {inst}")
        if not inst["pos"].strip() or not inst["neg"].strip():
            raise ValueError(f"Instruction pair {i} has an empty pos/neg field")

    questions = data["questions"]
    if not isinstance(questions, list):
        raise ValueError(f"Expected questions list, got {type(questions).__name__}")
    if len(questions) < EXPECTED_QUESTIONS:
        raise ValueError(
            f"Expected at least {EXPECTED_QUESTIONS} questions, got {len(questions)}. "
            "Re-run; gpt-4.1 occasionally undercounts."
        )
    if len(questions) > EXPECTED_QUESTIONS:
        # gpt-4.1 sometimes returns 41-42; trim deterministically. The split below
        # is seeded so this stays reproducible.
        print(f"  note: model returned {len(questions)} questions, trimming to first {EXPECTED_QUESTIONS}")
        del data["questions"][EXPECTED_QUESTIONS:]
        questions = data["questions"]
    for i, q in enumerate(questions):
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Question {i} is empty or non-string")

    if not isinstance(data["eval_prompt"], str) or not data["eval_prompt"].strip():
        raise ValueError("eval_prompt is missing or empty")


def _split_artifact(data: dict, seed: int = SEED) -> tuple[dict, dict]:
    """Split 40 questions into 20/20 deterministically (seed=42).
    Same instruction set + eval_prompt go in both halves."""
    rng = random.Random(seed)
    questions = list(data["questions"])
    rng.shuffle(questions)
    extract_questions = questions[: EXPECTED_QUESTIONS // 2]
    eval_questions = questions[EXPECTED_QUESTIONS // 2 :]

    common = {
        "instruction": data["instruction"],
        "eval_prompt": data["eval_prompt"],
    }
    extract_artifact = {**common, "questions": extract_questions}
    eval_artifact = {**common, "questions": eval_questions}
    return extract_artifact, eval_artifact


# ---------------------------------------------------------------------------
# Per-trait driver
# ---------------------------------------------------------------------------


def generate_for_trait(
    trait: str,
    description: str,
    *,
    client: OpenAI,
    model: str,
    overwrite: bool = False,
) -> bool:
    """Generate the extract+eval JSONs for one trait. Returns True if work was done,
    False if both files already existed and overwrite=False."""
    extract_path = EXTRACT_DIR / f"{trait}.json"
    eval_path = EVAL_DIR / f"{trait}.json"

    if not overwrite and extract_path.exists() and eval_path.exists():
        print(f"[skip] {trait}: both artifact files already exist")
        return False

    print(f"[generating] {trait}")
    prompt = _format_prompt(trait, description)
    raw = _call_openai(client, prompt, model)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{trait}: model returned invalid JSON ({e}). First 500 chars: {raw[:500]!r}"
        ) from e

    _validate_artifact(data)

    extract_artifact, eval_artifact = _split_artifact(data)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    extract_path.write_text(json.dumps(extract_artifact, indent=2, ensure_ascii=False))
    eval_path.write_text(json.dumps(eval_artifact, indent=2, ensure_ascii=False))
    print(f"  wrote {extract_path.relative_to(REPO_ROOT)}  ({len(extract_artifact['questions'])} questions)")
    print(f"  wrote {eval_path.relative_to(REPO_ROOT)}  ({len(eval_artifact['questions'])} questions)")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trait",
        default=None,
        help="Generate one specific trait. Default: all 8 in TRAIT_DEFINITIONS.",
    )
    p.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Generate this list of traits.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name (default: gpt-4.1)")
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if files exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.trait and args.traits:
        sys.exit("--trait and --traits are mutually exclusive")
    if args.trait:
        targets = [args.trait]
    elif args.traits:
        targets = args.traits
    else:
        targets = list(TRAIT_DEFINITIONS.keys())

    unknown = [t for t in targets if t not in TRAIT_DEFINITIONS]
    if unknown:
        sys.exit(f"Unknown traits: {unknown}. Known: {list(TRAIT_DEFINITIONS)}")

    client = OpenAI()
    print(f"model={args.model}  targets={targets}  overwrite={args.overwrite}")

    n_done = 0
    for trait in targets:
        try:
            if generate_for_trait(
                trait,
                TRAIT_DEFINITIONS[trait],
                client=client,
                model=args.model,
                overwrite=args.overwrite,
            ):
                n_done += 1
        except Exception as e:
            print(f"[FAIL] {trait}: {type(e).__name__}: {e}")
            # Continue with the remaining traits rather than aborting.

    print(f"\ndone — {n_done}/{len(targets)} traits generated this run")


if __name__ == "__main__":
    main()
