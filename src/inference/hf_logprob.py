"""
HF-flavored logprob delta for evaluating Anthropic-pipeline persona vectors against
MWE-format contrastive pairs.

Mirrors src/logprob.py compute_logprob_delta but uses raw HuggingFace transformers
+ the Anthropic-pipeline steering_hook (block-level forward hook), so the same
HF model loaded for run_validation_all.py LLM-judge stage can be reused without
re-loading via TransformerLens.

Layer indexing matches src/inference/hf_model.py steering_hook:
    layer_idx = i  ->  hook on model.model.layers[i]  ->  perturbs output of block i
                       which equals output_hidden_states[i + 1].
So a vector built at output_hidden_states[16] (Anthropic's "layer 16") is applied
with layer_idx=15.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from src.inference.hf_model import steering_hook


@torch.no_grad()
def compute_logprob_delta_hf(
    model,
    tokenizer,
    question: str,
    trait_completion: str,
    non_trait_completion: str,
    *,
    vector: torch.Tensor | None = None,
    layer_idx: int | None = None,
    alpha: float = 1.0,
) -> float:
    """
    Return log P(trait_completion | question, +alpha*v) - log P(non_trait_completion | question, +alpha*v).

    vector=None gives the unsteered baseline.  positions='all' on the steering hook
    so the perturbation applies across every token (matches src/logprob.py).
    """
    if not trait_completion or not non_trait_completion:
        return 0.0

    device = next(model.parameters()).device

    def _completion_logprob(full_text: str) -> float:
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
        q_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
        q_len = q_ids.shape[1]

        if vector is not None and layer_idx is not None and alpha != 0:
            ctx = steering_hook(model, vector, layer_idx=layer_idx, coeff=alpha, positions="all")
        else:
            ctx = nullcontext()

        with ctx:
            logits = model(full_ids).logits

        log_probs = torch.log_softmax(logits.float(), dim=-1)
        completion_ids = full_ids[0, q_len:]
        completion_log_probs = log_probs[0, q_len - 1 : -1, :]
        token_log_probs = completion_log_probs.gather(
            1, completion_ids.unsqueeze(1)
        ).squeeze(1)
        return token_log_probs.sum().item()

    lp_trait = _completion_logprob(question + trait_completion)
    lp_non = _completion_logprob(question + non_trait_completion)
    return lp_trait - lp_non
