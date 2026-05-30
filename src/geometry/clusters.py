# Antisocial cluster identified empirically from the L=16 cosine matrix.
# These traits form a positively-correlated block (pairwise cos +0.40 to
# +0.72) representing a shared manipulation / low-warmth / rude register
# subspace.
ANTISOCIAL_CLUSTER = frozenset({
    "evil", "impolite", "humorous",
    "power_seeking", "sycophantic", "apathetic",
})


def trait_cluster(trait: str) -> str:
    return "antisocial" if trait in ANTISOCIAL_CLUSTER else "other"


def pair_cluster_status(trait_i: str, trait_j: str) -> str:
    a, b = trait_i in ANTISOCIAL_CLUSTER, trait_j in ANTISOCIAL_CLUSTER
    if a and b:
        return "within_antisocial"
    if not a and not b:
        return "within_other"
    return "cross_cluster"
