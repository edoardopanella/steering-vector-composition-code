#!/bin/bash
#SBATCH --job-name=pilot-norm-modes
#SBATCH --output=/home/3247897/logs/validate_normalisations_pilot_%j.out
#SBATCH --error=/home/3247897/logs/validate_normalisations_pilot_%j.err
#SBATCH --time=08:00:00            # Pilot: 6 pairs × (1 baseline + 2 singles + 3 joints) × 20 q × N=5 = 3,600 generations. Phase 12 ran ~10h for ~24k gens; pro-rated ≤2h + model-load + buffer.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G                 # matches composition_scoring.sh (bf16 Llama-3.1-8B + HF generate)
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

# OPENAI_API_KEY (and anything else) lives in .env at repo root.
set -a; source .env; set +a

mkdir -p /home/3247897/logs

# Compute nodes also can't reach api.openai.com — generate stage only on the
# cluster. Judge stage runs on the laptop via validate_normalisations_judge_local.py
# (or composition_pilot's judge mode for the summary).
export COMPOSITION_PILOT_MODE=generate

echo "Starting validate_normalisations pilot — generate stage — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u -m scripts.compositions.validate_normalisations_pilot
echo "validate_normalisations pilot generate stage done — $(date)"
echo ""
echo "Next step (laptop, needs OPENAI_API_KEY):"
echo "  rsync results/composition_pilot_normalisations/Llama-3.1-8B-Instruct/*.csv from cluster"
echo "  python -m scripts.compositions.validate_normalisations_judge_local"
echo "  COMPOSITION_PILOT_MODE=judge python -m scripts.compositions.validate_normalisations_pilot"
