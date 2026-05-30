"""
Contrastive pair loading and splitting.

Each pair is a dict with keys "positive" and "negative".
On Mac/local: load from JSON files in data/.
On HPC: load from the MWE dataset format.

Split convention: 60% train (steering vector extraction),
                  20% val   (layer selection + coefficient tuning),
                  20% test  (all reported results).
"""

import importlib.util
import json
import random
from pathlib import Path


def load_contrastive_pairs(behavior: str, data_dir: str | Path) -> list[dict]:
    """Load contrastive pairs for a behavior.

    Accepts either <data_dir>/<behavior>.py (defines a module-level `pairs` list)
    or <data_dir>/<behavior>.json (list of {"positive": str, "negative": str}).
    """
    data_dir = Path(data_dir)
    py_path = data_dir / f"{behavior}.py"
    json_path = data_dir / f"{behavior}.json"

    if py_path.exists():
        spec = importlib.util.spec_from_file_location(f"_behavior_{behavior}", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(module.pairs)

    with open(json_path) as f:
        return json.load(f)


def split_pairs(
    pairs: list[dict],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split pairs into (train, val, test) with deterministic shuffle."""
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]
    return train, val, test


def save_pairs(pairs: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)


# Fixed neutral evaluation prompts.
# Used for layer selection and all composition experiments.
# Deliberately open-ended so the steering vector - not the prompt - elicits behavior.
EVAL_PROMPTS: list[str] = [
    "Tell me about something interesting you learned recently.",
    "What do you think about the way people use social media today?",
    "Can you help me understand what causes inflation?",
    "I'm thinking about changing careers. What should I consider?",
    "What's your view on the importance of sleep?",
    "How do you think cities could be better designed for people?",
    "Can you explain what machine learning actually is?",
    "What makes a good friend?",
    "I've been feeling a bit stuck lately. Any thoughts on how to move forward?",
    "What do you think about the future of remote work?",
    "How would you describe the difference between knowledge and wisdom?",
    "What are some things people often overlook when making decisions?",
    "Can you walk me through how vaccines work?",
    "What do you think is the most important skill someone can develop?",
    "How do languages shape the way we think?",
    "What should I know before starting to invest?",
    "Why do you think some people are more creative than others?",
    "What's the most interesting thing about how the human brain works?",
    "How do you think about balancing short-term and long-term goals?",
    "What makes a piece of writing truly memorable?",
]
