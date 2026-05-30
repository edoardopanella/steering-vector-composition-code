"""
Semantic similarity between traits via OpenAI embeddings.

For each of the 9 validated traits, we embed a one-sentence trait definition
and compute the 36 pairwise cosines. Result is the semantic-similarity covariate
used in the RQ1 logistic regression alongside |cos| of the steering vectors.

The trait descriptions are short, hand-written definitions (below) rather than
the judge eval prompts — the eval prompts are scoring-rubric text that mixes
the trait definition with rating-scale instructions and would shift the
embedding toward the meta-task, not the trait itself.

Output: results/semantic_similarity.json
    {
        "model": "text-embedding-3-large",
        "trait_text_source": "hand-written one-line definitions (semantic_sim.py)",
        "traits": [...],
        "trait_descriptions": {...},
        "sem_cos_matrix": [[...]],            # 9x9 cosine matrix, TRAITS order
        "pairs": [
            {"trait_a": "...", "trait_b": "...", "sem_sim": 0.42}, ...
        ]
    }

Run:
    python -m scripts.semantics.semantic_sim
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TRAITS = [
    # Tier S
    "apathetic", "evil", "hallucinating", "humorous", "impolite", "sycophantic",
    # Tier A
    "power_seeking", "confidence", "formality",
]

TRAIT_DESCRIPTIONS: dict[str, str] = {
    "apathetic":     "apathy: lack of interest, motivation, or emotional engagement; indifference to outcomes and to the user's concerns.",
    "confidence":    "confidence: assertiveness and certainty in one's statements; expressing claims without hedging or qualification.",
    "evil":          "evil: malicious intent, willingness to cause harm, and disregard for the wellbeing of others.",
    "formality":     "formality: use of formal language, professional register, and conventional structure; avoidance of slang or casual phrasing.",
    "hallucinating": "hallucination: fabrication of false or unsupported information presented as factual; inventing details that are not grounded in reality.",
    "humorous":      "humor: playful tone, jokes, wordplay, and levity; aiming to amuse or entertain the reader.",
    "impolite":      "impoliteness: rudeness, disrespect, and dismissiveness toward the user; lack of basic courtesy.",
    "power_seeking": "power-seeking: desire to acquire influence, control, resources, or autonomy; framing situations in terms of dominance and advantage.",
    "sycophantic":   "sycophancy: excessive agreement, flattery, and validation of the user regardless of accuracy; telling the user what they want to hear.",
}

EMBEDDING_MODEL = "text-embedding-3-large"

OUT_PATH = Path("results/semantic_similarity.json")


def main() -> None:
    missing = [t for t in TRAITS if t not in TRAIT_DESCRIPTIONS]
    if missing:
        raise RuntimeError(f"no description for traits: {missing}")
    texts = [TRAIT_DESCRIPTIONS[t] for t in TRAITS]
    print(f"Embedding {len(texts)} trait descriptions with {EMBEDDING_MODEL}")

    client = OpenAI()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    emb = np.array([r.embedding for r in resp.data], dtype=np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sem_cos = emb @ emb.T

    pairs = [
        {"trait_a": a, "trait_b": b,
         "sem_sim": float(sem_cos[TRAITS.index(a), TRAITS.index(b)])}
        for a, b in combinations(TRAITS, 2)
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "model": EMBEDDING_MODEL,
        "trait_text_source": "hand-written one-line definitions (semantic_sim.py)",
        "traits": TRAITS,
        "trait_descriptions": TRAIT_DESCRIPTIONS,
        "sem_cos_matrix": sem_cos.tolist(),
        "pairs": pairs,
    }, indent=2))

    print(f"Saved {len(pairs)} pair sims to {OUT_PATH}")
    print("\nTop 5 most semantically similar pairs:")
    for p in sorted(pairs, key=lambda r: -r["sem_sim"])[:5]:
        print(f"  {p['trait_a']:<14} {p['trait_b']:<14} {p['sem_sim']:+.3f}")
    print("\nBottom 5 (most dissimilar):")
    for p in sorted(pairs, key=lambda r: r["sem_sim"])[:5]:
        print(f"  {p['trait_a']:<14} {p['trait_b']:<14} {p['sem_sim']:+.3f}")


if __name__ == "__main__":
    main()
