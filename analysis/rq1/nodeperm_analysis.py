"""
nodeperm_analysis.py

Runner for the RQ1 per-scheme decomposition with the exact 8! node-permutation
(MRQAP) test. Just run this file directly - no terminal arguments needed.

It reads the consolidated (coh≥30) frame, fits the per-scheme regression
`y ~ mean_single_abs + max_single_abs + cos` for the **norm** (`normTrue`) and
**per-axis** (`per_axis`) schemes, and recomputes β_cos's p-value with the
exhaustive dyadic node-permutation null. It writes one combined report:

    analysis/results/RQ1/perscheme_nodeperm_analysis/perscheme_nodeperm_analysis.{md,json}

The `## Conclusions` section of the .md is preserved across re-runs.

Script port of analysis/notebooks/rq1_perscheme_nodeperm.ipynb.
"""

from nodeperm_analysis_scripts import run_nodeperm_analysis

# --------------------------------------------------------------------------- #
# Parameters - edit these, then run the file.
# --------------------------------------------------------------------------- #
N_EDGE_PERM = 10000     # Monte-Carlo iters for the edge-wise comparison in §2
                        # (the node-perm test itself is always exhaustive, n!)


if __name__ == "__main__":
    run_nodeperm_analysis(n_edge_perm=N_EDGE_PERM)
