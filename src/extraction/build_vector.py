"""
Build persona vectors from the (pos, neg) extract CSVs.

Direct port of anthropic_code/generate_vec.py:
    1. Load both CSVs.
    2. Effectiveness + coherence filter:
         pos.trait >= 50, neg.trait < 50, both coherence >= 50.
    3. For each surviving (prompt, answer): forward-pass prompt+answer through the
       model with output_hidden_states=True and accumulate three quantities per layer:
         - prompt_avg[L]:   mean over prompt tokens of hidden_states[L]
         - response_avg[L]: mean over response tokens of hidden_states[L]   <-- paper's primary
         - prompt_last[L]:  hidden state at the last prompt token
    4. Mean-difference (pos minus neg), per layer, NOT normalised.
    5. Save three [N_layers+1, hidden_dim] tensors.

Indexing note: hidden_states has length num_hidden_layers + 1 (index 0 = embedding
output, indices 1..N = post-block residual stream after block i-1). To steer at
the equivalent point with our forward-hook approach, build vector_at_layer[L] from
output_hidden_states[L] and inject at hf_model.steering_hook(layer_idx = L - 1).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def filter_effective(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    trait: str,
    threshold: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anthropic's get_persona_effective filter. Both frames must be aligned 1:1."""
    if len(pos_df) != len(neg_df):
        raise ValueError(
            f"pos and neg CSVs must have matching row counts (got {len(pos_df)} vs {len(neg_df)})"
        )
    mask = (
        (pos_df[trait] >= threshold)
        & (neg_df[trait] < (100 - threshold))
        & (pos_df["coherence"] >= threshold)
        & (neg_df["coherence"] >= threshold)
    )
    return pos_df[mask].reset_index(drop=True), neg_df[mask].reset_index(drop=True)


@torch.no_grad()
def collect_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    layer_list: list[int] | None = None,
):
    """
    For each (prompt, answer): tokenize prompt+answer concatenated, single forward pass,
    record per-layer prompt_avg / response_avg / prompt_last.

    Returns three lists, one per layer in layer_list, each a tensor of shape
    [n_examples, hidden_dim]. CPU, float32.
    """
    n_layers = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(n_layers + 1))  # match Anthropic's full stack

    prompt_avg = [[] for _ in range(n_layers + 1)]
    response_avg = [[] for _ in range(n_layers + 1)]
    prompt_last = [[] for _ in range(n_layers + 1)]

    device = next(model.parameters()).device

    for prompt, answer in tqdm(zip(prompts, answers), total=len(prompts), desc="hidden_states"):
        text = prompt + answer
        # add_special_tokens=False because the chat template already includes them.
        full = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        prompt_len = prompt_ids.shape[0]

        out = model(**full, output_hidden_states=True)
        hs = out.hidden_states  # tuple of length n_layers+1, each [1, T, hidden]

        for L in layer_list:
            h = hs[L]  # [1, T, hidden]
            prompt_avg[L].append(h[:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[L].append(h[:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[L].append(h[:, prompt_len - 1, :].detach().cpu())

        del out

    for L in layer_list:
        prompt_avg[L] = torch.cat(prompt_avg[L], dim=0).float()
        response_avg[L] = torch.cat(response_avg[L], dim=0).float()
        prompt_last[L] = torch.cat(prompt_last[L], dim=0).float()

    return prompt_avg, response_avg, prompt_last


def build_persona_vectors(
    *,
    model_name: str,
    pos_csv: Path,
    neg_csv: Path,
    trait: str,
    save_dir: Path,
    threshold: int = 50,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """
    Replicates anthropic_code/generate_vec.py end-to-end.
    Returns a dict with the three resulting tensors (also written to disk).
    """
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    print(f"loaded pos: {len(pos_df)} rows  neg: {len(neg_df)} rows")

    pos_eff, neg_eff = filter_effective(pos_df, neg_df, trait, threshold=threshold)
    print(f"after filter (pos>={threshold}, neg<{100-threshold}, coh>={threshold}): {len(pos_eff)} pairs")
    if len(pos_eff) == 0:
        raise RuntimeError(
            "Filter dropped every pair. Check that the extract step actually elicited "
            "the trait under positive instructions and not under negative."
        )

    print(f"loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pos_p_avg, pos_r_avg, pos_p_last = collect_hidden_states(
        model, tokenizer, pos_eff["prompt"].tolist(), pos_eff["answer"].tolist()
    )
    neg_p_avg, neg_r_avg, neg_p_last = collect_hidden_states(
        model, tokenizer, neg_eff["prompt"].tolist(), neg_eff["answer"].tolist()
    )

    n_layers = model.config.num_hidden_layers
    prompt_avg_diff = torch.stack(
        [pos_p_avg[L].mean(0) - neg_p_avg[L].mean(0) for L in range(n_layers + 1)], dim=0
    )
    response_avg_diff = torch.stack(
        [pos_r_avg[L].mean(0) - neg_r_avg[L].mean(0) for L in range(n_layers + 1)], dim=0
    )
    prompt_last_diff = torch.stack(
        [pos_p_last[L].mean(0) - neg_p_last[L].mean(0) for L in range(n_layers + 1)], dim=0
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_avg_diff, save_dir / f"{trait}_prompt_avg_diff.pt")
    torch.save(response_avg_diff, save_dir / f"{trait}_response_avg_diff.pt")
    torch.save(prompt_last_diff, save_dir / f"{trait}_prompt_last_diff.pt")
    print(f"saved 3 vectors of shape {tuple(response_avg_diff.shape)} to {save_dir}")

    return {
        "prompt_avg_diff": prompt_avg_diff,
        "response_avg_diff": response_avg_diff,
        "prompt_last_diff": prompt_last_diff,
        "n_effective": len(pos_eff),
    }
