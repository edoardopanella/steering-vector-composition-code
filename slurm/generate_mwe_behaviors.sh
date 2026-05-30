#!/bin/bash
#SBATCH --job-name=generate-mwe-behaviors
#SBATCH --output=/home/3242106/logs/generate_mwe_behaviors_%j.out
#SBATCH --error=/home/3242106/logs/generate_mwe_behaviors_%j.err
#SBATCH --time=02:00:00         # ~352 OpenAI calls × ~3s each ≈ 18 min; 2h cushion for retries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2       # pure API job, no compute
#SBATCH --mem=8G                # only holds JSON in memory; 8 traits × 1000 pairs is small
#SBATCH --qos=stud
#SBATCH --account=3242106
#SBATCH --partition=stud
#SBATCH --chdir=/home/3242106/steering-vector-composition-cloned

set -euo pipefail

module purge
module load miniconda3
source activate steering-vector-composition-venv

# OPENAI_API_KEY lives in .env
set -a; source .env; set +a

mkdir -p /home/3242106/logs
mkdir -p data/behaviors_mwe

echo "Starting generate-mwe-behaviors — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/generate_mwe_behaviors.py
echo "generate-mwe-behaviors done — $(date)"

# Final verification step (auto-runs after generation)
echo
echo "=== Verification ==="
python -u -c "
from src.datasets import load_contrastive_pairs, split_pairs
TRAITS = ['apathetic','evil','humorous','impolite','optimistic','refusal','sycophantic','hallucinating']
for t in TRAITS:
    try:
        p = load_contrastive_pairs(t, 'data/behaviors_mwe')
        tr, va, te = split_pairs(p)
        print(f'{t:<14} n={len(p)}  train/val/test={len(tr)}/{len(va)}/{len(te)}')
    except Exception as e:
        print(f'{t:<14} FAIL: {e}')
"
