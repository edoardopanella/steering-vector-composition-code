"""
Local LLM-judge stage for composition scoring - laptop entrypoint.

Single purpose: walk the unordered (trait_a, trait_b) pairs over the validated
traits; for each pair, judge the 4 per-setting CSVs (baseline, single_a,
single_b, joint) that the cluster generate stage left with NaN score columns.
Fills trait_a / trait_b / coherence in place. Nothing else.

Post-judge analysis (regime, L17 sanity, aggregate trajectory parquet, tau
calibration, summary JSON) is the job of composition_scoring.py in judge
mode - run it after this script:

    COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring

That second pass sees every CSV already scored, skips re-judging, and goes
straight to the aggregate + τ + summary work.

Designed for laptop:
  - no GPU, no HF model load
  - needs OPENAI_API_KEY in .env and outbound internet
  - idempotent on row.trait_a.isna() - kill + rerun is safe

Data needed on laptop (rsync from cluster before running):
  results/composition/unnormalized_sum_a4/scoring/Llama-3.1-8B-Instruct/*.csv
  data/composition_eval/*.json

Run:
    python -m scripts.compositions.composition_judge_local
"""
from __future__ import annotations

from dotenv import load_dotenv

from scripts.compositions.composition_scoring import (
    COMPOSITION_ALPHA,
    JUDGE_MODEL,
    LOGS_DIR,
    SCORES_OUTPUT_DIR,
    _baseline_csv_path,
    _composition_pairs,
    _csv_has_scores,
    _csv_status,
    _judge_csv_inplace,
    _load_composition_artifact,
    _single_csv_path,
    _steered_csv_path,
)

load_dotenv()


def _per_pair_settings(a: str, b: str):
    """(label, csv_path, log_filename, judge_progress_tag) for the four
    settings - naming matches composition_scoring.py so judge-stage log lines
    accumulate in the same per-pair files left by the generate stage."""
    return [
        (
            "baseline",
            _baseline_csv_path(a, b),
            f"composition_{a}__{b}_baseline.log",
            f"{a}+{b} baseline",
        ),
        (
            "single_a",
            _single_csv_path(a, b, "a", COMPOSITION_ALPHA),
            f"composition_{a}__{b}_single_a_alpha{COMPOSITION_ALPHA}.log",
            f"{a}+{b} single_a α={COMPOSITION_ALPHA}",
        ),
        (
            "single_b",
            _single_csv_path(a, b, "b", COMPOSITION_ALPHA),
            f"composition_{a}__{b}_single_b_alpha{COMPOSITION_ALPHA}.log",
            f"{a}+{b} single_b α={COMPOSITION_ALPHA}",
        ),
        (
            "joint",
            _steered_csv_path(a, b, COMPOSITION_ALPHA),
            f"composition_{a}__{b}_alpha{COMPOSITION_ALPHA}.log",
            f"{a}+{b} joint α={COMPOSITION_ALPHA}",
        ),
    ]


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LOCAL JUDGE STAGE")
    print(f"  judge model : {JUDGE_MODEL}")
    print(f"  CSV dir     : {SCORES_OUTPUT_DIR}")
    print(f"  per-pair logs append to {LOGS_DIR}/composition_<a>__<b>_*.log")
    print("  idempotent on row.trait_a.isna() - safe to kill + rerun")
    print("  for regime / parquet / τ run composition_scoring.py mode=judge after")
    print("=" * 72 + "\n")

    pairs = _composition_pairs()
    print(f"Judging {len(pairs)} composition pairs at α={COMPOSITION_ALPHA}\n")

    for i, (a, b) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {a} + {b}")
        try:
            artifact = _load_composition_artifact(a, b)
        except FileNotFoundError as e:
            print(f"  skipping - {e}")
            continue

        for label, csv_path, log_name, tag in _per_pair_settings(a, b):
            print(f"  {label:<9} -> {csv_path.name}")
            if not csv_path.exists():
                print(f"    SKIP: CSV missing (generate stage not done on cluster)")
                continue
            _judge_csv_inplace(
                csv_path,
                artifact["eval_prompt_a"], artifact["eval_prompt_b"],
                progress_tag=tag,
                log_path=LOGS_DIR / log_name,
            )
            print(f"    {_csv_status(csv_path)}    log={log_name}")

    # End-of-run tally - only counts CSVs scored, no other artifacts touched.
    n_csvs = len(list(SCORES_OUTPUT_DIR.glob("*.csv")))
    n_scored = sum(1 for p in SCORES_OUTPUT_DIR.glob("*.csv") if _csv_has_scores(p))
    print()
    print("=" * 72)
    print("JUDGE STAGE TALLY")
    print(f"  CSVs on disk          : {n_csvs}  (expected 144)")
    print(f"  CSVs with judge scores: {n_scored}  (target 144)")
    print("=" * 72)
    print(
        "\nNext step: build aggregate parquet + τ + summary JSON by running\n"
        "  COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring"
    )


if __name__ == "__main__":
    main()
