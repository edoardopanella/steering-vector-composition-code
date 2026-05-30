import ast
from pathlib import Path

import pandas as pd

CSV_PATH = Path("results/human_eval/human_eval_layer17.csv")
XLSX_PATH = Path("results/human_eval/human_eval_layer17.xlsx")


def parse_pair(x):
    if isinstance(x, tuple):
        return x
    return ast.literal_eval(x)


def parse_setting(x):
    if isinstance(x, tuple):
        return x
    return ast.literal_eval(x)


df = pd.read_csv(CSV_PATH)

pairs = df["behavior_pair"].map(parse_pair)
settings = df["setting"].map(parse_setting)

out = pd.DataFrame({
    "behavior_1": pairs.map(lambda p: p[0]),
    "behavior_2": pairs.map(lambda p: p[1]),
    "coef_1": settings.map(lambda s: s[0]),
    "coef_2": settings.map(lambda s: s[1]),
    "prompt": df["prompt"],
    "completion": df["completion"],
    "judge_b1": df["judge_b1"],
    "judge_b2": df["judge_b2"],
    "judge_coherence": df["judge_coherence"],
    "rating_b1": df.get("rating_b1", ""),
    "rating_b2": df.get("rating_b2", ""),
    "notes": df.get("notes", ""),
})

XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_excel(XLSX_PATH, index=False)
print(f"Wrote {len(out)} rows to {XLSX_PATH}")
