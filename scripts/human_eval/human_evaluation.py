import asyncio
import json
from pathlib import Path

import pandas as pd

from src.datasets import EVAL_PROMPTS
from src.composition.human_samples import sample_completions
from src.extraction.generation import COHERENCE_PROMPT, _judge_all
from src.judge import OpenAiJudge

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 600
TEMPERATURE = 0.7

LAYER = 17
VECTORS_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct/")
ALPHA = 4.0

EVAL_PROMPTS = EVAL_PROMPTS
N_PROMPTS = 20

# LLM-judge config - same convention as composition_scoring.
JUDGE_MODEL = "gpt-4.1-mini"
MAX_CONCURRENT_JUDGES = 5
COMPOSITION_DATA_DIR = Path("data/composition_eval")

BEHAVIORS = [
    "formality",
    "confidence",
    "evil",
]

SETTING = [((0,0), 2),
           ((1,0), 4),
           ((0,1), 4),
           ((1,1), 6),
           ((-1,1), 2),
           ((1,-1), 2),
           ]

OUT_DIR = Path("results/human_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pair_eval_prompts(behavior_pair):
    """Return (eval_prompt_b1, eval_prompt_b2) aligned to behavior_pair order.

    The composition_eval artifact stores eval_prompt_a / eval_prompt_b under the
    sorted (a, b) trait order; remap them to the pair order used in the data.
    """
    a, b = sorted(behavior_pair)
    artifact = json.loads((COMPOSITION_DATA_DIR / f"{a}__{b}.json").read_text())
    by_trait = {a: artifact["eval_prompt_a"], b: artifact["eval_prompt_b"]}
    return by_trait[behavior_pair[0]], by_trait[behavior_pair[1]]


def judge_completions(df):
    """Score each completion 0-100 on its two traits + coherence. Adds columns
    judge_b1, judge_b2, judge_coherence in place and returns df. Human rating
    columns are left untouched for human-vs-machine comparison.
    """
    def _to_nan(xs):
        return [float("nan") if x is None else float(x) for x in xs]

    df["judge_b1"] = pd.array([pd.NA] * len(df), dtype="Float64")
    df["judge_b2"] = pd.array([pd.NA] * len(df), dtype="Float64")
    df["judge_coherence"] = pd.array([pd.NA] * len(df), dtype="Float64")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        judge_coh = OpenAiJudge(JUDGE_MODEL, COHERENCE_PROMPT, eval_type="0_100")
        for behavior_pair, group in df.groupby("behavior_pair", sort=False):
            eval_b1, eval_b2 = load_pair_eval_prompts(behavior_pair)
            judge_b1 = OpenAiJudge(JUDGE_MODEL, eval_b1, eval_type="0_100")
            judge_b2 = OpenAiJudge(JUDGE_MODEL, eval_b2, eval_type="0_100")

            questions = group["prompt"].astype(str).tolist()
            answers = group["completion"].astype(str).fillna("").tolist()
            tag = f"{behavior_pair[0]}+{behavior_pair[1]}"
            print(f"\n=== Judging pair {tag} ({len(questions)} completions) ===")

            scores_b1 = loop.run_until_complete(_judge_all(
                judge_b1, questions, answers, MAX_CONCURRENT_JUDGES,
                progress_label=f"{tag} b1", progress_every=25,
            ))
            scores_b2 = loop.run_until_complete(_judge_all(
                judge_b2, questions, answers, MAX_CONCURRENT_JUDGES,
                progress_label=f"{tag} b2", progress_every=25,
            ))
            scores_coh = loop.run_until_complete(_judge_all(
                judge_coh, questions, answers, MAX_CONCURRENT_JUDGES,
                progress_label=f"{tag} coherence", progress_every=25,
            ))

            df.loc[group.index, "judge_b1"] = _to_nan(scores_b1)
            df.loc[group.index, "judge_b2"] = _to_nan(scores_b2)
            df.loc[group.index, "judge_coherence"] = _to_nan(scores_coh)
    finally:
        loop.close()
    return df


if __name__ == "__main__":

    data = sample_completions(
        model_name=MODEL,
        device=DEVICE,
        layer=LAYER,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        vectors_dir=VECTORS_DIR,
        behaviors=BEHAVIORS,
        alpha=ALPHA,
        normalize=True,
        settings=SETTING,
        n_prompts=N_PROMPTS,
        eval_prompts=EVAL_PROMPTS,
    )

    df = pd.DataFrame(data, columns=["behavior_pair", "setting", "prompt", "completion"])

    # Machine judge scores (0-100).
    df = judge_completions(df)

    # Empty columns for human annotators (kept separate from judge_* scores).
    df["rating_b1"] = ""
    df["rating_b2"] = ""
    df["notes"] = ""

    out_path = OUT_DIR / f"human_eval_layer{LAYER}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
