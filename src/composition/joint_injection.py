"""
Steering vector composition + injection on raw HuggingFace transformers.

Joint:  h^(L) <- h^(L) + alpha * direction
where `direction` is built by `compose_steering_vector` from per-vector weights
(unit-normalised then re-normalised) and `alpha` is the calibrated injection
magnitude (E10.3/E10.4 protocol).

Generation reuses `src.extraction.generation.generate_batch` so the joint
pipeline runs on the same HF + forward-hook code path as the validated alpha
sweep at L=17.
"""

import torch


def compose_steering_vector(
    vectors_weights: list[tuple[torch.Tensor, float]],
    alpha: float,
    normalize: bool | str = True,
) -> torch.Tensor:
    """Build a steering vector from per-vector weights at injection scale `alpha`.

    Three composition modes (see paper/experiments_log.md §E15.2):

      normalize=False   δ = α · Σ w_i v_i
        Coefficient on each vector is held constant at α. Both per-axis push
        and total ‖δ‖ scale with the geometry of the inputs. Phase 12 default.

      normalize=True    δ = α · (Σ w_i v̂_i) / ‖Σ w_i v̂_i‖
        Total ‖δ‖ held constant at α. Per-axis push shrinks at non-zero cos.

      normalize="per_axis"
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
        # weaker axis below α — closer to the single-injection regime than
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

    `layer` follows the Anthropic-pipeline convention (persona-stack index =
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
# Helpers for RQ2 — Phase 1: projection pipeline (per RQ2 roadmap)
#
# Operating point fixed in E10.4: L* = 17, α_unit = 4, unit-normalised
# response_avg_diff[17] vectors. Eq (1): π = ⟨h^(L), v^(L*)⟩ / ‖v^(L*)‖.
# Eq (2): Δ = mean_L |π^(1,1)(L) - π^(1,0)(L)|. Eq (3): L_div = first L
# crossing τ. Trajectories are response-token-averaged per the upstream
# extraction convention (build_vector.collect_hidden_states).
# =========================================================================

from pathlib import Path

PERSONA_VECTOR_DIR = Path(
    "results/persona_vectors/Llama-3.1-8B-Instruct"
)


def load_unit_vector(trait: str, layer: int = 17) -> torch.Tensor:
    """Load response_avg_diff[layer] for `trait` and unit-normalise.

    Matches the E10.3 / E10.4 protocol: slice the [33, 4096] stack at the
    operating layer and divide by ‖v‖. Returns float32 on CPU.
    """
    path = PERSONA_VECTOR_DIR / f"{trait}_response_avg_diff.pt"
    full_stack = torch.load(path, map_location="cpu", weights_only=True)
    v = full_stack[layer].float()
    n = v.norm()
    if n == 0:
        raise ValueError(f"zero-norm vector at layer {layer} for {trait}")
    return v / n


def project_activation(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Eq (1): π = ⟨h, v⟩ / ‖v‖. `h` shape (..., hidden), `v` shape (hidden,).

    Robust to v non-unit even though E10 vectors are unit-normalised — the
    explicit ‖v‖ keeps the primitive correct if the pipeline ever reverts to
    raw vectors (per roadmap §2 closing note).
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
    positions of block (L*-1)'s output — i.e. the same residual point the
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


def project_trajectory(
    activations: dict[int, torch.Tensor],
    v: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Apply eq (1) layer-by-layer. `activations[L]` has shape (..., hidden)
    (typically [B, hidden] when stacked across prompts, or [hidden,] for a
    single sample). Returns dict[L] -> projection with the leading shape
    preserved.
    """
    return {L: project_activation(h, v) for L, h in activations.items()}


def stack_trajectory(
    per_prompt: list[dict[int, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Stack per-prompt trajectory dicts into [n_prompts, hidden] per layer.

    Each input dict shares the same layer set; returns one tensor per layer.
    """
    if not per_prompt:
        return {}
    layers = sorted(per_prompt[0].keys())
    return {
        L: torch.stack([d[L] for d in per_prompt], dim=0)
        for L in layers
    }


def traj_divergence(
    pi_joint: dict[int, torch.Tensor],
    pi_indiv: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Eq (2): Δ_i(i, j) = (1/|L|) Σ_L |π^(1,1)(L) - π^(1,0)(L)|.

    Pass projected trajectories. Per-prompt average over layers; returns a
    tensor with the leading shape of the projection values (scalar for a
    single sample, [B] when projections are batched).
    """
    layers = sorted(pi_joint.keys())
    if set(layers) != set(pi_indiv.keys()):
        raise ValueError("trajectories must share the same layer set")
    diffs = torch.stack(
        [(pi_joint[L] - pi_indiv[L]).abs() for L in layers], dim=0
    )
    return diffs.mean(dim=0)


def layer_of_divergence(
    pi_joint: dict[int, torch.Tensor],
    pi_indiv: dict[int, torch.Tensor],
    tau: float,
    layer_final: int | None = None,
) -> torch.Tensor:
    """Eq (3): L_div = min{L ∈ L : |π^(1,1)(L) - π^(1,0)(L)| > τ}.

    Trajectories that never cross τ saturate at `layer_final` (default = max
    layer in the trajectory). Returns a long tensor with the projection's
    leading shape.
    """
    layers = sorted(pi_joint.keys())
    if set(layers) != set(pi_indiv.keys()):
        raise ValueError("trajectories must share the same layer set")
    if layer_final is None:
        layer_final = layers[-1]

    diffs = torch.stack(
        [(pi_joint[L] - pi_indiv[L]).abs() for L in layers], dim=0
    )  # [n_L, ...]
    mask = diffs > tau                         # [n_L, ...]
    any_cross = mask.any(dim=0)                # [...]
    first_idx = mask.long().argmax(dim=0)      # [...] — 0 if never True
    layer_tensor = torch.tensor(
        layers, dtype=torch.long, device=mask.device
    )
    picked = layer_tensor[first_idx]
    return torch.where(any_cross, picked, torch.full_like(picked, layer_final))


def calibrate_tau(
    pi_indiv_trajectories: list[dict[int, torch.Tensor]],
    factor: float = 1.5,
) -> float:
    """Phase 1 pilot: τ = `factor` × max layer-to-layer noise observed across
    individual-steering trajectories. Per roadmap §4 ("calibrate τ as 1.5×
    the maximum layer-to-layer noise observed in the individual-steering
    condition, and pre-register it").
    """
    max_step = 0.0
    for traj in pi_indiv_trajectories:
        layers = sorted(traj.keys())
        for a, b in zip(layers, layers[1:]):
            step = (traj[b] - traj[a]).abs().max().item()
            if step > max_step:
                max_step = step
    return factor * max_step


# --- synthetic smoke tests (roadmap §4 "Unit testing") -------------------

def _smoke_tests() -> None:
    """Run the four primitives on synthetic data where the answer is known.
    No model required. Raises AssertionError on regression.
    """
    # 1. Projection along the basis recovers the magnitude.
    v = torch.tensor([1.0, 0.0, 0.0])
    h_aligned = torch.tensor([3.7, 0.0, 0.0])
    assert abs(project_activation(h_aligned, v).item() - 3.7) < 1e-6

    # 2. Orthogonal -> 0.
    h_ortho = torch.tensor([0.0, 5.0, 0.0])
    assert abs(project_activation(h_ortho, v).item()) < 1e-6

    # 3. Non-unit v: ‖v‖ in the denominator makes the projection scale-invariant
    # in v — projecting h onto v and onto k·v gives the same number.
    v_big = 4.0 * v
    h_big = torch.tensor([8.0, 0.0, 0.0])
    pi_unit = project_activation(h_big, v).item()
    pi_big = project_activation(h_big, v_big).item()
    assert abs(pi_unit - pi_big) < 1e-6
    assert abs(pi_unit - 8.0) < 1e-6

    # 4. Δ matches the average of a known drift signal.
    pi_a = {17: torch.tensor(0.0), 18: torch.tensor(1.0), 19: torch.tensor(2.0)}
    pi_b = {17: torch.tensor(0.0), 18: torch.tensor(0.0), 19: torch.tensor(0.0)}
    assert abs(traj_divergence(pi_a, pi_b).item() - 1.0) < 1e-6

    # 5. L_div picks the first layer crossing τ.
    pi_j = {17: torch.tensor(0.0), 18: torch.tensor(0.0),
            19: torch.tensor(2.0), 20: torch.tensor(3.0)}
    pi_i = {17: torch.tensor(0.0), 18: torch.tensor(0.0),
            19: torch.tensor(0.0), 20: torch.tensor(0.0)}
    assert layer_of_divergence(pi_j, pi_i, tau=1.0).item() == 19

    # 6. L_div saturates at layer_final when no layer crosses τ.
    assert layer_of_divergence(pi_j, pi_i, tau=10.0, layer_final=99).item() == 99

    # 7. Batched: leading shape preserved through the pipeline.
    pi_j_b = {17: torch.tensor([0.0, 0.0]),
              18: torch.tensor([0.5, 1.0]),
              19: torch.tensor([1.5, 2.0])}
    pi_i_b = {17: torch.tensor([0.0, 0.0]),
              18: torch.tensor([0.0, 0.0]),
              19: torch.tensor([0.0, 0.0])}
    delta = traj_divergence(pi_j_b, pi_i_b)
    assert torch.allclose(delta, torch.tensor([2.0 / 3, 3.0 / 3]), atol=1e-6)
    Ldiv = layer_of_divergence(pi_j_b, pi_i_b, tau=0.4)
    assert Ldiv.tolist() == [18, 18]

    # --- compose_steering_vector mode coverage (E15.2) -------------------
    import math
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2_pos = torch.tensor([math.cos(math.pi / 3), math.sin(math.pi / 3), 0.0])  # cos=+0.5
    v2_neg = torch.tensor([math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3), 0.0])  # cos=-0.5
    alpha = 4.0

    # Singles collapse identically across modes (one weight = 0).
    for mode in (False, True, "per_axis"):
        d = compose_steering_vector([(v1, 1.0), (v2_pos, 0.0)], alpha=alpha, normalize=mode)
        assert torch.allclose(d, alpha * v1, atol=1e-6), f"single failed in mode {mode}"

    # normalize=False: δ = α·(v̂_1 + v̂_2), per-axis push = α·(1+cos).
    d_false = compose_steering_vector(
        [(v1, 1.0), (v2_pos, 1.0)], alpha=alpha, normalize=False
    )
    assert abs(float((d_false @ v1).item()) - alpha * 1.5) < 1e-5

    # normalize=True: ‖δ‖ = α exactly.
    d_true = compose_steering_vector(
        [(v1, 1.0), (v2_pos, 1.0)], alpha=alpha, normalize=True
    )
    assert abs(d_true.norm().item() - alpha) < 1e-5

    # per_axis: per-axis push onto each input = α; ‖δ‖ = α·√(2/(1+cos)).
    for v2, cos_val in [(v2_pos, 0.5), (v2_neg, -0.5)]:
        d_pa = compose_steering_vector(
            [(v1, 1.0), (v2, 1.0)], alpha=alpha, normalize="per_axis"
        )
        push_1 = float((d_pa @ v1).item())
        push_2 = float((d_pa @ (v2 / v2.norm())).item())
        assert abs(push_1 - alpha) < 1e-5, f"per_axis push_1={push_1} ≠ α at cos={cos_val}"
        assert abs(push_2 - alpha) < 1e-5, f"per_axis push_2={push_2} ≠ α at cos={cos_val}"
        expected_norm = alpha * math.sqrt(2.0 / (1.0 + cos_val))
        assert abs(d_pa.norm().item() - expected_norm) < 1e-4, (
            f"per_axis ‖δ‖={d_pa.norm().item()} ≠ {expected_norm} at cos={cos_val}"
        )

    # per_axis at cos=0 reduces to normalize=False (no rescaling needed).
    v_ortho = torch.tensor([0.0, 1.0, 0.0])
    d_pa_ortho = compose_steering_vector(
        [(v1, 1.0), (v_ortho, 1.0)], alpha=alpha, normalize="per_axis"
    )
    d_false_ortho = compose_steering_vector(
        [(v1, 1.0), (v_ortho, 1.0)], alpha=alpha, normalize=False
    )
    assert torch.allclose(d_pa_ortho, d_false_ortho, atol=1e-6)

    # 8. τ calibration picks 1.5 × max step in the individual trajectories.
    indiv = [
        {17: torch.tensor(0.0), 18: torch.tensor(0.1), 19: torch.tensor(0.3)},
        {17: torch.tensor(0.0), 18: torch.tensor(0.05), 19: torch.tensor(0.15)},
    ]
    tau = calibrate_tau(indiv, factor=1.5)
    assert abs(tau - 1.5 * 0.2) < 1e-6   # max step is 0.3 - 0.1 = 0.2

    print("smoke tests passed.")


if __name__ == "__main__":
    _smoke_tests()


