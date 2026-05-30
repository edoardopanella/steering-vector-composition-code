'''
Human-evaluation code for steering vector composability analysis.
Parameters:
- BEHAVIORS: sample of 3 behaviors from each of the 3 categories (safety, style, persona);
- LAYER: single layer number selected for injection based on scores;
'''

from pathlib import Path

import torch
import random

from src.inference.hf_model import load_hf_model
from src.composition.joint_injection import apply_steering_batched, compose_steering_vector
from src.composition.joint_behaviors import behavior_pairs


def sample_completions(
    model_name: str, device: str, layer: int, max_new_tokens: int, temperature: float,
    vectors_dir: Path, behaviors: list, alpha: float, normalize: bool,
    settings: list[tuple[tuple[int, int], int]], n_prompts: int, eval_prompts: list[str],
    batch_size: int = 8,
) -> list[tuple[tuple[str, str], tuple[int, int], str, str]]:

    # --- Sanity check ---
    missing = [b for b in behaviors if not (vectors_dir / f"{b}_response_avg_diff.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing vector files for behaviors: {missing}")

    if sum(t[1] for t in settings) != n_prompts:
        raise ValueError(f"Total number of completions per setting must equal {n_prompts} (currently {sum(t[1] for t in settings)})")

    if len(eval_prompts) < n_prompts:
        raise ValueError(f"eval_prompts has only {len(eval_prompts)} entries, need {n_prompts}")

    print(f"Loading model: {model_name}")
    model, tokenizer = load_hf_model(model_name)
    model_device = next(model.parameters()).device

    pairs = behavior_pairs(behaviors)
    print(f"\n=== Layer {layer} ===")

    data = []
    for behavior_pair in pairs:
        print(f"\n=== Pair {behavior_pair} ===")
        vector1 = torch.load(vectors_dir / f"{behavior_pair[0]}_response_avg_diff.pt", weights_only=True)[layer].to(model_device)
        vector2 = torch.load(vectors_dir / f"{behavior_pair[1]}_response_avg_diff.pt", weights_only=True)[layer].to(model_device)

        prompts_shuffled = random.sample(eval_prompts, n_prompts)
        n_prompt = 0

        for setting, n in settings:
            print(f"\n=== Setting {setting} ===")
            vectors_weights = [(vector1, setting[0]), (vector2, setting[1])]
            steering_vector = compose_steering_vector(vectors_weights, alpha=alpha, normalize=normalize)

            assigned_prompts = prompts_shuffled[n_prompt:n_prompt + n]
            n_prompt += n

            completions = apply_steering_batched(
                model=model, tokenizer=tokenizer,
                prompts=assigned_prompts, layer=layer,
                steering_vector=steering_vector,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
            for prompt, completion in zip(assigned_prompts, completions):
                data.append((behavior_pair, setting, prompt, completion))

    print("Done.")
    return data
