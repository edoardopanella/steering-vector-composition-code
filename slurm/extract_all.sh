#!/bin/bash
#SBATCH --job-name=anthropic-repl-extract-all
#SBATCH --output=/home/3247897/logs/extract_all_%j.out
#SBATCH --error=/home/3247897/logs/extract_all_%j.err
#SBATCH --time=23:59:00         # ~40 min/trait × 12 remaining traits ≈ 8h; max student walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G              # bumped from 128G; HF + Llama-bf16 only needs ~16-20G but we have headroom on stud nodes and want zero risk over an 8h run
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3247897
#SBATCH --partition=stud
#SBATCH --chdir=/home/3247897/steering-vector-composition

set -euo pipefail

export SCRATCH=/mnt/beegfsstudents/home/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

mkdir -p /home/3247897/logs

echo "Starting anthropic-repl extract-all — $(date)"
export PYTHONPATH=/home/3247897/steering-vector-composition
python -u -m scripts.extraction.run_extract_all
echo "Anthropic-repl extract-all done — $(date)"
