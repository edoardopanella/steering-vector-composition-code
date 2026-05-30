"""
Pilot: laptop judge stage for the normalisation comparison.

Walks every CSV in the pilot output directory and judges any rows with NaN
trait_a. Picks up CSVs from both pilot 1 (the original 36) and pilot 2 (18
new joint conditions + any future additions) automatically - the loop is
filename-driven, not condition-list-driven, so dropping new CSVs into the
dir just works.

Post-judge tally + per-pair summary table is the job of
validate_normalisations_pilot.py / validate_normalisations_pilot_2.py in
judge mode - run those after this script.

Designed for laptop:
  - no GPU, no HF model load
  - needs OPENAI_API_KEY in .env and outbound internet
  - idempotent on row.trait_a.isna() - kill + rerun is safe

Data needed on laptop (rsync from cluster before running):
  results/pilots/composition_normalisations/Llama-3.1-8B-Instruct/*.csv
  data/composition_eval/*.json

Run:
    python -m scripts.compositions.validate_normalisations_judge_local
"""
from __future__ import annotations

from dotenv import load_dotenv

import re

from scripts.compositions.validate_normalisations_pilot import (
    JUDGE_MODEL,
    LOGS_DIR,
    SCORES_OUTPUT_DIR,
    _csv_has_scores,
    _csv_status,
    _judge_csv_inplace,
    _load_composition_artifact,
)

load_dotenv()

# Filename patterns:
#   {a}__{b}_baseline.csv
#   {a}__{b}_single_{which}_alpha{α}.csv
#   {a}__{b}_joint_{mode_tag}_alpha{α}.csv
# NB: trait names may contain underscores (e.g. "hallucinating"), so anchor
# `rest` to one of the known suffix shapes rather than relying on a lazy
# wildcard between b and rest.
_FILENAME_RE = re.compile(
    r"^(?P<a>[a-z_]+?)__(?P<b>[a-z_]+?)"
    r"_(?P<rest>(?:baseline|single_[ab]_alpha[0-9.]+|joint_(?:false|true|per_axis)_alpha[0-9.]+))"
    r"\.csv$"
)


def _parse_csv(path):
    """Return (a, b, label, log_name, progress_tag) or None if not parseable.

    `label` is short ("baseline" / "single_a" / "joint_per_axis"); the
    log_name + progress_tag follow the pilot 1 conventions so the same
    per-pair log files accumulate across pilots.
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    a, b, rest = m["a"], m["b"], m["rest"]
    # rest is one of:
    #   "baseline"
    #   "single_<which>_alpha<α>"
    #   "joint_<mode_tag>_alpha<α>"
    if rest == "baseline":
        return a, b, "baseline", f"pilot_{a}__{b}_baseline.log", f"{a}+{b} baseline"
    if rest.startswith("single_"):
        # single_a_alpha4.0 or single_b_alpha4.0
        return a, b, rest, f"pilot_{a}__{b}_{rest}.log", f"{a}+{b} {rest}"
    if rest.startswith("joint_"):
        return a, b, rest, f"pilot_{a}__{b}_{rest}.log", f"{a}+{b} {rest}"
    return None


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("LOCAL JUDGE STAGE - pilot normalisation comparison")
    print(f"  judge model : {JUDGE_MODEL}")
    print(f"  CSV dir     : {SCORES_OUTPUT_DIR}")
    print("  walks every CSV in the dir; picks up pilot 1 + pilot 2 + future")
    print("  idempotent on row.trait_a.isna() - safe to kill + rerun")
    print("=" * 72 + "\n")

    csvs = sorted(SCORES_OUTPUT_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {SCORES_OUTPUT_DIR}")
        return

    # Group by pair so we load each composition artifact once.
    by_pair: dict[tuple[str, str], list] = {}
    skipped: list[str] = []
    for p in csvs:
        info = _parse_csv(p)
        if info is None:
            skipped.append(p.name)
            continue
        a, b, label, log_name, tag = info
        by_pair.setdefault((a, b), []).append((p, label, log_name, tag))

    if skipped:
        print(f"WARN: {len(skipped)} unparseable filenames: {skipped[:3]}{'...' if len(skipped) > 3 else ''}")

    print(f"Judging {len(csvs)} CSVs across {len(by_pair)} pairs\n")

    for i, ((a, b), entries) in enumerate(sorted(by_pair.items()), 1):
        print(f"\n[{i}/{len(by_pair)}] {a} + {b}  ({len(entries)} CSVs)")
        try:
            artifact = _load_composition_artifact(a, b)
        except FileNotFoundError as e:
            print(f"  skipping - {e}")
            continue
        for csv_path, label, log_name, tag in sorted(entries):
            print(f"  {label:<26} -> {csv_path.name}")
            _judge_csv_inplace(
                csv_path,
                artifact["eval_prompt_a"], artifact["eval_prompt_b"],
                progress_tag=tag,
                log_path=LOGS_DIR / log_name,
            )
            print(f"    {_csv_status(csv_path)}    log={log_name}")

    n_csvs = len(list(SCORES_OUTPUT_DIR.glob("*.csv")))
    n_scored = sum(1 for p in SCORES_OUTPUT_DIR.glob("*.csv") if _csv_has_scores(p))
    print()
    print("=" * 72)
    print("JUDGE STAGE TALLY")
    print(f"  CSVs on disk          : {n_csvs}")
    print(f"  CSVs with judge scores: {n_scored}")
    print("=" * 72)
    print(
        "\nNext step: build per-pair summary tables\n"
        "  COMPOSITION_PILOT_MODE=judge python -m scripts.compositions.validate_normalisations_pilot\n"
        "  COMPOSITION_PILOT_MODE=judge python -m scripts.compositions.validate_normalisations_pilot_2"
    )


if __name__ == "__main__":
    main()
