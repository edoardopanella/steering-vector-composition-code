"""
Steering vector composition + injection on raw HuggingFace transformers.

Joint:  h^(L) <- h^(L) + alpha * direction
where `direction` is built by `compose_steering_vector` from per-vector weights
(unit-normalised then re-normalised) and `alpha` is the calibrated injection
magnitude.

Generation reuses `src.extraction.generation.generate_batch` so the joint
pipeline runs on the same HF + forward-hook code path as the single-vector
alpha sweep at the operating layer L=17.
"""

import torch


def compose_steering_vector(
    vectors_weights: list[tuple[torch.Tensor, float]],
    alpha: float,
    normalize: bool | str = True,
) -> torch.Tensor:
    """Build a steering vector from per-vector weights at injection scale `alpha`.

    Three composition modes:

      normalize=False   (unnormalised sum)   δ = α · Σ w_i v_i
        Coefficient on each vector is held constant at α. Both per-axis push
        and total ‖δ‖ scale with the geometry of the inputs.

      normalize=True    (normalised sum)     δ = α · (Σ w_i v̂_i) / ‖Σ w_i v̂_i‖
        Total ‖δ‖ held constant at α. Per-axis push shrinks at non-zero cos.

      normalize="per_axis"  (projection-controlled)
        Unit-normalise inputs, sum on the unit sphere, then rescale so that
        each non-zero contributor sees a per-axis projection push of exactly α
        (matching what it would receive under single-vector injection at α).
        For two vectors with both weights = 1 and cosine c this reduces to
            δ = (α / (1 + c)) · (v̂_i + v̂_j),
        giving per-axis push = α (constant) and total ‖δ‖ = α·√(2/(1+c)).
        Singles ((1,0)/(0,1)) are identical to the other modes.

    The non-zero-weights guard keeps singles handled correctly: with one
    weight = 0 the formulation collapses to α·v̂_active in every mode.
    """
    if normalize is False:
        return alpha * sum(w * v for v, w in vectors_weights)

    unit_terms = [w * (v / v.norm()) for v, w in vectors_weights if v.norm() > 0]
    if not unit_terms:
        return torch.zeros_like(vectors_weights[0][0])
    direction = sum(unit_terms)

    if normalize is True:
        dn = direction.norm()
        if dn == 0:
            return torch.zeros_like(direction)
        return alpha * (direction / dn)

    if normalize == "per_axis":
        # Per-axis projection onto each active unit vector v̂_k is
        #   ⟨direction, v̂_k⟩ = w_k + Σ_{j≠k} w_j cos(v̂_k, v̂_j).
        # Rescaling by `scale` produces per-axis push = scale · that quantity.
        # We pick the largest of those terms as the reference push and rescale
        # so it equals α. This generalises the two-vector case (where both
        # per-axis pushes equal 1+cos and scale = α/(1+cos) trivially):
        # in the asymmetric case it caps the dominant axis at α and leaves the
        # weaker axis below α - closer to the single-injection regime than
        # over-shooting it.
        active_units = [
            (w, v / v.norm()) for v, w in vectors_weights
            if v.norm() > 0 and w != 0
        ]
        if not active_units:
            return torch.zeros_like(direction)
        per_axis_pushes = [
            float((direction @ v_hat).item()) for _, v_hat in active_units
        ]
        ref = max(abs(p) for p in per_axis_pushes)
        if ref == 0:
            return torch.zeros_like(direction)
        return (alpha / ref) * direction

    raise ValueError(
        f"normalize must be True, False, or 'per_axis'; got {normalize!r}"
    )


def apply_steering_batched(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    steering_vector: torch.Tensor,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> list[str]:
    """Batched steered generation with a precomposed steering vector.

    `layer` follows the persona-stack convention (persona-stack index =
    `hidden_states[layer]`). The forward hook fires on the output of
    transformer block `layer-1`, so the perturbation lands in the same residual
    stream the vectors were extracted from.

    A zero-norm steering vector (e.g. setting=(0,0)) installs no hook, giving
    a clean unsteered baseline.
    """
    from src.extraction.generation import generate_batch

    conversations = [[{"role": "user", "content": p}] for p in prompts]
    if steering_vector.norm().item() > 0:
        steering = (steering_vector, layer - 1, 1.0, "response")
    else:
        steering = None
    _, answers = generate_batch(
        model, tokenizer, conversations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        steering=steering,
    )
    return answers


# =========================================================================
# Projection pipeline.
#
# Operating point: L* = 17, unit-normalised response_avg_diff[17] vectors.
# Projection of an activation onto a steering vector is
#   π = ⟨h^(L), v^(L*)⟩ / ‖v^(L*)‖.
# Trajectories are response-token-averaged per the upstream extraction
# convention (build_vector.collect_hidden_states).
# =========================================================================


def project_activation(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Projection π = ⟨h, v⟩ / ‖v‖. `h` shape (..., hidden), `v` shape (hidden,).

    Robust to v non-unit even though the operating-point vectors are
    unit-normalised - the explicit ‖v‖ keeps the primitive correct if the
    pipeline ever reverts to raw (unnormalised) vectors.
    """
    h32 = h.float()
    v32 = v.float().reshape(-1)
    vn = v32.norm()
    if vn == 0:
        return torch.zeros(h32.shape[:-1], dtype=h32.dtype, device=h32.device)
    return (h32 @ v32) / vn


@torch.no_grad()
def trajectory_response_avg(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    layers_above: list[int],
    delta_at_lstar: torch.Tensor | None = None,
    layer_star: int = 17,
) -> dict[int, torch.Tensor]:
    """Teacher-forced forward pass on (prompt + answer) with an optional
    additive perturbation `delta_at_lstar` injected at the response-token
    positions of block (L*-1)'s output - i.e. the same residual point the
    composition sweep steers. Returns response-averaged activations for each
    layer L in `layers_above`.

    Convention matches build_vector.collect_hidden_states and
    hf_model.steering_hook: hidden_states[L] = output of block L-1, so the
    L*=17 hook fires on model.model.layers[16].

    Returns dict[L] -> tensor [hidden] on CPU float32. Pass delta=None for the
    unsteered baseline.
    """
    from src.inference.hf_model import _resolve_layer_list

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    layer_list = _resolve_layer_list(model)

    text = prompt + answer
    full = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
    prompt_len = prompt_ids.shape[0]

    handles = []
    if delta_at_lstar is not None:
        delta = delta_at_lstar.to(dtype=dtype, device=device).reshape(-1)
        if delta.numel() != model.config.hidden_size:
            raise ValueError(
                f"delta has {delta.numel()} dims, expected hidden_size "
                f"{model.config.hidden_size}"
            )

        def _steer_hook(_module, _inputs, outputs):
            hs = outputs[0] if isinstance(outputs, tuple) else outputs
            rest = outputs[1:] if isinstance(outputs, tuple) else None
            new_hs = hs.clone()
            new_hs[:, prompt_len:, :] = new_hs[:, prompt_len:, :] + delta
            return (new_hs, *rest) if rest is not None else new_hs

        handles.append(
            layer_list[layer_star - 1].register_forward_hook(_steer_hook)
        )

    try:
        out = model(**full, output_hidden_states=True)
        hs_stack = out.hidden_states  # tuple length n_layers+1

        avg: dict[int, torch.Tensor] = {}
        for L in layers_above:
            h = hs_stack[L][0, prompt_len:, :]  # [T_resp, hidden]
            if h.shape[0] == 0:
                avg[L] = torch.zeros(model.config.hidden_size)
            else:
                avg[L] = h.mean(dim=0).float().cpu()
    finally:
        for h in handles:
            h.remove()

    return avg
