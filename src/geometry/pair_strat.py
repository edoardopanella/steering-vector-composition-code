import numpy as np
import pandas as pd

from src.geometry.clusters import pair_cluster_status, trait_cluster

NEAR_MAX = 0.2
MODERATE_MAX = 0.35


def get_offdiag(G: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    n = G.shape[0]
    idx = np.triu_indices(n, k=1)
    return G[idx], idx


def assign_stratum(abs_cos: float) -> str:
    if abs_cos < NEAR_MAX:
        return "near"
    if abs_cos < MODERATE_MAX:
        return "moderate"
    return "high"


def make_pairs_df(G: np.ndarray, behavior_order: list[str]) -> pd.DataFrame:
    values, idx = get_offdiag(G)
    rows, cols = idx
    behavior_i = [behavior_order[i] for i in rows]
    behavior_j = [behavior_order[j] for j in cols]
    abs_values = np.abs(values)
    return pd.DataFrame({
        "i": rows,
        "j": cols,
        "behavior_i": behavior_i,
        "behavior_j": behavior_j,
        "cosine": values,
        "abs_cosine": abs_values,
        "stratum": [assign_stratum(v) for v in abs_values],
        "trait_i_cluster": [trait_cluster(t) for t in behavior_i],
        "trait_j_cluster": [trait_cluster(t) for t in behavior_j],
        "pair_cluster_status": [pair_cluster_status(a, b)
                                for a, b in zip(behavior_i, behavior_j)],
        "both_antisocial": [pair_cluster_status(a, b) == "within_antisocial"
                            for a, b in zip(behavior_i, behavior_j)],
    })


def stratify_pairs(
    pairs_df: pd.DataFrame,
    n_near: int | None = None,
    n_moderate: int | None = None,
    n_high: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Default behaviour: keep all pairs in every stratum (no downsampling).
    Pass an explicit `n_*` to downsample a stratum for a balanced composition
    pair-pick. With defaults, output covers the full population from `pairs_df`."""

    near = pairs_df[pairs_df["abs_cosine"] < NEAR_MAX]
    moderate = pairs_df[(pairs_df["abs_cosine"] >= NEAR_MAX) &
                        (pairs_df["abs_cosine"] < MODERATE_MAX)]
    high = pairs_df[pairs_df["abs_cosine"] >= MODERATE_MAX]

    def _sample(bin_df, n_target, label):
        if n_target is None:
            return bin_df
        if len(bin_df) < n_target:
            print(f"[WARN] stratum '{label}' has {len(bin_df)} pairs, wanted {n_target}")
            n_target = len(bin_df)
        return bin_df.sample(n=n_target, random_state=seed)

    near_sample = _sample(near, n_near, "near").assign(stratum="near")
    moderate_sample = _sample(moderate, n_moderate, "moderate").assign(stratum="moderate")
    high_sample = _sample(high, n_high, "high").assign(stratum="high")

    return pd.concat([near_sample, moderate_sample, high_sample]).reset_index(drop=True)
