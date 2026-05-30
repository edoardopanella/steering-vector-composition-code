"""
Joint judge for evaluating completions produced by joint steering injection.

For each (behavior_pair, setting, prompt, completion) record, scores the
completion independently against:
  - behavior 1  (0-100, BEHAVIOR_PROMPTS in src.scoring)
  - behavior 2  (0-100, BEHAVIOR_PROMPTS in src.scoring)
  - coherence   (0-100, Anthropic-style COHERENCE_PROMPT)

All three scores share the OpenAiJudge logprob-aggregation path. Coherence is
the same prompt used at extraction/validation time, so the >=50 keeper
threshold from src.extraction.generation carries over.

Returns a DataFrame keyed on (behavior_pair, setting, prompt) so it can be
merged directly with the human-eval annotation frame.
"""

import asyncio

import pandas as pd

from src.extraction.generation import COHERENCE_PROMPT
from src.judge import OpenAiJudge
from src.scoring import BEHAVIOR_PROMPTS, JUDGE_MODEL


def _make_behavior_judges(behaviors: list[str], judge_model: str) -> dict[str, OpenAiJudge]:
    return {
        b: OpenAiJudge(judge_model, BEHAVIOR_PROMPTS[b], eval_type="0_100")
        for b in behaviors
    }


async def _score_row(row, behavior_judges, coherence_judge):
    pair, setting, prompt, completion = row
    b1, b2 = pair
    score_b1, score_b2, coherence = await asyncio.gather(
        behavior_judges[b1](question=prompt, answer=completion),
        behavior_judges[b2](question=prompt, answer=completion),
        coherence_judge(question=prompt, answer=completion),
    )
    return (pair, setting, prompt, completion, score_b1, score_b2, coherence)


async def _score_all(data, behavior_judges, coherence_judge, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def bounded(row):
        async with sem:
            return await _score_row(row, behavior_judges, coherence_judge)

    return await asyncio.gather(*[bounded(r) for r in data])


def score_joint_completions(
    data: list[tuple[tuple[str, str], tuple[int, int], str, str]],
    judge_model: str = JUDGE_MODEL,
    concurrency: int = 16,
) -> pd.DataFrame:
    """Score each completion against both behaviors of its pair plus coherence.

    Args:
        data: rows from sample_completions — (pair, setting, prompt, completion).
        judge_model: OpenAI model for the judge.
        concurrency: max in-flight API calls.

    Returns DataFrame with columns:
        behavior_pair, setting, prompt, completion, score_b1, score_b2, coherence
    """
    behaviors = sorted({b for pair, _, _, _ in data for b in pair})
    missing = [b for b in behaviors if b not in BEHAVIOR_PROMPTS]
    if missing:
        raise KeyError(f"No BEHAVIOR_PROMPTS entry for: {missing}")

    behavior_judges = _make_behavior_judges(behaviors, judge_model)
    coherence_judge = OpenAiJudge(judge_model, COHERENCE_PROMPT, eval_type="0_100")

    scored = asyncio.run(
        _score_all(data, behavior_judges, coherence_judge, concurrency=concurrency)
    )
    return pd.DataFrame(
        scored,
        columns=[
            "behavior_pair", "setting", "prompt", "completion",
            "score_b1", "score_b2", "coherence",
        ],
    )
