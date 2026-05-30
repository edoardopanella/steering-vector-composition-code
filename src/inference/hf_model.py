"""
HuggingFace model loader and steering hook for the Anthropic-pipeline replication.

Deliberately NOT TransformerLens. The Anthropic pipeline relies on three things that
are easier on raw HF transformers:
  - output_hidden_states=True returning the full [n_layers+1, ...] stack in one pass
  - tokenizer.apply_chat_template (Qwen / Llama-3 / etc., handled by AutoTokenizer)
  - register_forward_hook on model.model.layers[L] for response-only steering

Llama-3.1-8B-Instruct: hidden_size=4096, num_hidden_layers=32 -> output_hidden_states
returns 33 tensors (index 0 = embeddings, indices 1..32 = post-block residual streams).
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _log(msg: str) -> None:
    """Stage print that also flushes - process stdout is line-buffered with -u
    but `redirect_stdout` capture (used elsewhere) can mask flushes, so emit
    to the real fd directly with a timestamp."""
    print(f"[hf_model {time.strftime('%H:%M:%S')}] {msg}", file=sys.__stdout__, flush=True)


def _resolve_local_snapshot(model_name: str) -> str:
    """Resolve `model_name` to a local cached snapshot path.

    Why: transformers >= 4.46 calls `model_info(model_id)` inside the tokenizer
    loader (`_patch_mistral_regex`) to detect base-Mistral checkpoints. That
    metadata call has no cache fallback, so on a networkless compute node it
    hard-fails (errno 101) - or worse, waits for a TCP timeout that never
    arrives because the cluster firewall drops outbound packets without RST.
    Passing a local path triggers the `_is_local` short-circuit, which skips
    the metadata lookup entirely.

    Behaviour:
      * Cache hit -> return local snapshot directory.
      * Cache miss / corrupt -> raise loudly (callers must NOT silently fall
        back to the bare repo id on a networkless node; that's what caused the
        previous "stuck on Loading ..." hang).
    """
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    try:
        path = snapshot_download(repo_id=model_name, local_files_only=True)
        _log(f"snapshot resolved: {path}")
        return path
    except Exception as e:
        if offline:
            raise RuntimeError(
                f"HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE is set but no local snapshot "
                f"of {model_name!r} was found: {type(e).__name__}: {e}. "
                f"Pre-download the model on a node with internet:\n"
                f"  HF_HUB_OFFLINE=0 python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; "
                f"AutoTokenizer.from_pretrained({model_name!r}); "
                f"AutoModelForCausalLM.from_pretrained({model_name!r})\""
            ) from e
        _log(f"snapshot_download(local_files_only=True) failed ({e}); falling back to repo id")
        return model_name


def load_hf_model(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    _log(f"resolving snapshot for {model_name} (HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')})")
    src = _resolve_local_snapshot(model_name)
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        _log(f"cuda: device={torch.cuda.get_device_name(0)}  free={free/1e9:.1f}G / total={total/1e9:.1f}G")
    else:
        _log("cuda: NOT available - model will load on CPU")
    _log(f"from_pretrained: weights, dtype={dtype}, device_map={device_map}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=dtype, device_map=device_map
    )
    _log(f"weights loaded in {time.time() - t0:.1f}s")
    model.eval()
    _log("from_pretrained: tokenizer")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(src)
    _log(f"tokenizer loaded in {time.time() - t0:.1f}s")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return model, tok


def _resolve_layer_list(model: torch.nn.Module):
    # Llama / Mistral
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Qwen / fallback
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Could not locate the transformer block list on the model. "
        "Expected model.model.layers (Llama) or model.transformer.h (Qwen/GPT2)."
    )


@contextmanager
def steering_hook(
    model: torch.nn.Module,
    vector: torch.Tensor,
    layer_idx: int,
    coeff: float = 1.0,
    positions: str = "response",
    prompt_len: int | None = None,
) -> Iterator[None]:
    """
    Register a forward hook on transformer block `layer_idx` that adds
    `coeff * vector` to the block's output residual stream.

    positions:
        "all"      - every token position (used during teacher-forced extraction).
        "response" - last token only. During autoregressive decoding this is the
                     newly-generated token at every step, so the cumulative effect
                     is "perturb every response token". Matches Anthropic's
                     ActivationSteerer(positions="response") for inference.
        "prompt"   - every token in the prefill. Requires prompt_len for masked steering.

    The hook fires on the *output* of model.model.layers[layer_idx], which equals
    `hidden_states[layer_idx + 1]` from output_hidden_states=True. So if you build a
    vector from output_hidden_states[L], inject it at layer_idx = L - 1.
    """
    layers = _resolve_layer_list(model)
    if not (-len(layers) <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(layers)} layers")
    if positions not in {"all", "response", "prompt"}:
        raise ValueError(f"positions must be one of all/response/prompt, got {positions!r}")

    p = next(model.parameters())
    v = vector.to(dtype=p.dtype, device=p.device).reshape(-1)
    if v.numel() != model.config.hidden_size:
        raise ValueError(
            f"Vector length {v.numel()} != hidden_size {model.config.hidden_size}"
        )
    delta = coeff * v

    def _hook(_module, _inputs, outputs):
        # Llama returns a tuple (hidden_states, ...).
        if isinstance(outputs, tuple):
            hs = outputs[0]
            rest = outputs[1:]
        else:
            hs = outputs
            rest = None

        if positions == "all":
            new_hs = hs + delta.to(hs.device)
        elif positions == "response":
            # During decoding hs.shape[1] == 1 after the first prefill step;
            # write into the last position uniformly.
            new_hs = hs.clone()
            new_hs[:, -1, :] += delta.to(hs.device)
        else:  # "prompt"
            if prompt_len is None:
                raise RuntimeError("positions='prompt' requires prompt_len")
            new_hs = hs.clone()
            new_hs[:, :prompt_len, :] += delta.to(hs.device)

        return (new_hs, *rest) if rest is not None else new_hs

    handle = layers[layer_idx].register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
