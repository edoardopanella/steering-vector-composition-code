#!/bin/bash
#SBATCH --job-name=composition-scoring-v2
#SBATCH --output=/home/3247897/logs/composition_scoring_v2_%j.out
#SBATCH --error=/home/3247897/logs/composition_scoring_v2_%j.err
#SBATCH --time=23:59:00          # Phase 12.5: 28 pairs × (baseline + 2 singles + joint) gens + 28 × ~9600 teacher-force traces. Phase 12 (36 pairs) used ~11h; pro-rate ~8.5h + cushion.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3247897
#SBATCH --partition=stud
#SBATCH --chdir=/home/3247897/steering-vector-composition
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=riccardo.scibetta7@gmail.com

set -euo pipefail

export SCRATCH=/mnt/beegfsstudents/home/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/torch_cache

# Compute nodes have no outbound network. Force HF Hub offline so:
#  (1) from_pretrained skips HEAD revalidation and uses cached files,
#  (2) transformers' _patch_mistral_regex skips model_info() metadata call
#      (which has no cache fallback and would otherwise hard-fail).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

mkdir -p /home/3247897/logs

# normalized_sum scheme (paper Eq 2). Generate stage only on cluster; judge
# stage runs on laptop with the same COMPOSITION_SCHEME:
#   COMPOSITION_SCHEME=normalized_sum COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring
export COMPOSITION_SCHEME=normalized_sum
export COMPOSITION_MODE=generate

echo "Starting normalized-sum composition scoring (normalize=True, α=4.5, 28 pairs) — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u -m scripts.compositions.composition_scoring
echo "Normalized-sum generate stage done — $(date)"
echo ""
echo "Next step (laptop, needs OPENAI_API_KEY):"
echo "  # 1) Pull new CSVs (use --ignore-existing to protect any local scores):"
echo "  rsync -av --ignore-existing bocconi-hpc:/home/3247897/steering-vector-composition/results/composition/normalized_sum_a4.5/ ./results/composition/normalized_sum_a4.5/"
echo "  # 2) Judge + aggregate + τ + summary:"
echo "  COMPOSITION_SCHEME=normalized_sum COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring"
