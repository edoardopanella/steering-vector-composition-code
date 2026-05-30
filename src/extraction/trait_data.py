"""
Loader for Anthropic persona-vectors trait artifacts.

Each trait JSON has the schema:
    {
        "instruction": [{"pos": str, "neg": str}, ...],   # 5 system-prompt pairs
        "questions":   [str, ...],                         # 20 trait-eliciting questions
        "eval_prompt": str                                 # judge rubric template
    }

Files live under anthropic_code/data_generation/trait_data_{extract,eval}/{trait}.json
in the upstream repo we vendored. We read them directly - no copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIT_DATA_DIR = REPO_ROOT / "external" / "anthropic_code" / "data_generation"


@dataclass
class TraitArtifact:
    trait: str
    instructions: list[dict]   # list of {"pos": str, "neg": str}
    questions: list[str]
    eval_prompt: str


def load_trait(trait: str, version: str = "extract") -> TraitArtifact:
    """version: "extract" (vector-building set) or "eval" (held-out evaluation set)."""
    if version not in {"extract", "eval"}:
        raise ValueError(f"version must be 'extract' or 'eval', got {version!r}")
    path = TRAIT_DATA_DIR / f"trait_data_{version}" / f"{trait}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Trait artifact not found at {path}. "
            f"Available traits: {sorted(p.stem for p in (TRAIT_DATA_DIR / f'trait_data_{version}').glob('*.json'))}"
        )
    data = json.loads(path.read_text())
    return TraitArtifact(
        trait=trait,
        instructions=data["instruction"],
        questions=data["questions"],
        eval_prompt=data["eval_prompt"],
    )


def a_or_an(word: str) -> str:
    return "an" if word and word[0].lower() in "aeiou" else "a"


def system_prompt(assistant_name: str, instruction: str) -> str:
    """Anthropic's exact format: 'You are a/an {name} assistant. {instruction}'."""
    return f"You are {a_or_an(assistant_name)} {assistant_name} assistant. {instruction}"
