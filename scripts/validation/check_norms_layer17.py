"""
Quick diagnostic - vector norms at L=17 for every extracted trait, plus the
correlation between norm and alpha=2 effect size on the validated keepers.

Reads the response_avg_diff stacks saved by run_extract_all and the alpha-sweep
summary saved by run_validation_all_layer17.

Run locally (no GPU, no API):
    venv/bin/python scripts/validation/check_norms_layer17.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

VECTOR_DIR = Path("results/persona_vectors/Llama-3.1-8B-Instruct")
SUMMARY_L17 = Path("results/validation_summary_layer17.json")
HIDDEN_LAYER = 17


def collect_norms() -> dict[str, float]:
    norms: dict[str, float] = {}
    for f in sorted(VECTOR_DIR.glob("*_response_avg_diff.pt")):
        trait = f.stem.replace("_response_avg_diff", "")
        v = torch.load(f, map_location="cpu", weights_only=True)
        if v.ndim != 2 or v.shape[0] <= HIDDEN_LAYER:
            print(f"  WARNING: {f.name} unexpected shape {tuple(v.shape)} - skipping")
            continue
        norms[trait] = v[HIDDEN_LAYER].norm().item()
    return norms


def collect_effects() -> dict[str, dict[str, float]]:
    """Per-trait (Δ_trait @ α=2, lp_shift @ α=2) from the L=17 α-sweep summary."""
    if not SUMMARY_L17.exists():
        return {}
    with open(SUMMARY_L17) as f:
        s = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for r in s["traits"]:
        if r["status"] != "ok":
            continue
        j = r["llm_judge"]["alphas"]["2.0"]
        lp = r["logprob"]["alphas"]["2.0"]
        out[r["trait"]] = {
            "delta_trait": j["delta_trait"],
            "lp_shift": lp["mean_shift"],
        }
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float:
    def _ranks(arr: list[float]) -> list[float]:
        s = sorted(range(len(arr)), key=lambda i: arr[i])
        rk = [0.0] * len(arr)
        for r, i in enumerate(s, 1):
            rk[i] = r
        return rk
    return pearson(_ranks(xs), _ranks(ys))


def main() -> None:
    norms = collect_norms()
    effects = collect_effects()

    items = sorted(norms.items(), key=lambda kv: -kv[1])
    n_min = min(norms.values())
    n_max = max(norms.values())
    n_med = sorted(norms.values())[len(norms) // 2]

    print(f"=== L={HIDDEN_LAYER} norms ‖v‖₂ for {len(norms)} traits ===")
    print(f"{'trait':<18} {'‖v‖':>8} {'α≈2/‖v‖':>9}  {'judge Δ@α=2':>11}  {'logprob@α=2':>11}")
    print("-" * 65)
    for t, n in items:
        e = effects.get(t)
        if e is None:
            jdt, lps = "    -", "    -"
        else:
            jdt = f"{e['delta_trait']:+11.2f}"
            lps = f"{e['lp_shift']:+11.4f}"
        # α-equivalent: raw α=2 ≈ unit α = 2·‖v‖ - show inverse so readers see "raw α=2 ≡ unit α=X".
        unit_alpha_equiv = 2.0 * n
        print(f"{t:<18} {n:>8.3f} {unit_alpha_equiv:>9.3f}  {jdt}  {lps}")

    print()
    print(f"min ‖v‖ = {n_min:.3f}   max ‖v‖ = {n_max:.3f}   median ‖v‖ = {n_med:.3f}   spread max/min = {n_max/n_min:.2f}×")

    if effects:
        # Correlations on the 9 validated keepers.
        joint_traits = sorted(set(norms) & set(effects))
        ns = [norms[t] for t in joint_traits]
        jd = [effects[t]["delta_trait"] for t in joint_traits]
        lp = [effects[t]["lp_shift"] for t in joint_traits]
        print()
        print(f"=== correlations (n={len(joint_traits)}) ===")
        print(f"  Pearson  ‖v‖ × judge Δ@α=2   = {pearson(ns, jd):+.3f}")
        print(f"  Spearman ‖v‖ × judge Δ@α=2   = {spearman(ns, jd):+.3f}")
        print(f"  Pearson  ‖v‖ × logprob@α=2   = {pearson(ns, lp):+.3f}")
        print(f"  Spearman ‖v‖ × logprob@α=2   = {spearman(ns, lp):+.3f}")
        print()
        print("Interpretation:")
        print("  |r| > 0.5  -> norm dominance is real, unit-normalisation justified for composition.")
        print("  |r| < 0.2  -> norm spread is incidental; raw α=2 may be safe for composition.")
        print("  intermediate -> mixed signal; per-trait α calibration recommended either way.")
    else:
        print(f"\n[no validation_summary_layer17.json at {SUMMARY_L17}; skipping correlation block]")


if __name__ == "__main__":
    main()
