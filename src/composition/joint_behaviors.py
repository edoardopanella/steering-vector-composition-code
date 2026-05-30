from itertools import combinations

def behavior_pairs(behaviors: list[str]) -> list[tuple[str, str]]:
    """Return all unique (unordered) pairs from a list of behavior names."""
    return list(combinations(behaviors, 2))
