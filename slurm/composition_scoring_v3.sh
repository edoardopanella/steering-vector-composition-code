#!/bin/bash
#SBATCH --job-name=composition-scoring-v3
#SBATCH --output=/home/3247897/logs/composition_scoring_v3_%j.out
#SBATCH --error=/home/3247897/logs/composition_scoring_v3_%j.err
#SBATCH --time=23:59:00          # Phase 15.16 (per_axis): 28 pairs at α=4.5 per_axis. Same gen+trace budget as Phase 12.5 (~9h); cushion for slowdown.
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

# Generate stage only on cluster. Judge stage runs on laptop:
#   COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring_v3
export COMPOSITION_MODE=generate

echo "Starting Phase 15.16 (per_axis) (composition_scoring_v3: normalize=per_axis, α=4.5, 28 pairs) — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u -m scripts.compositions.composition_scoring_v3
echo "Phase 15.16 (per_axis) generate stage done — $(date)"
echo ""
echo "Next step (laptop, needs OPENAI_API_KEY):"
echo "  # 1) Pull new CSVs (use --ignore-existing to protect any local scores):"
echo "  rsync -av --ignore-existing bocconi-hpc:/home/3247897/steering-vector-composition/results/composition_scoring_l17_v3/ ./results/composition_scoring_l17_v3/"
echo "  rsync -av --ignore-existing bocconi-hpc:/home/3247897/steering-vector-composition/results/composition_trajectories_l17_v3/ ./results/composition_trajectories_l17_v3/"
echo "  # 2) Judge + aggregate + τ + summary:"
echo "  COMPOSITION_MODE=judge python -m scripts.compositions.composition_scoring_v3"
