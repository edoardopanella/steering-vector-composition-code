#!/bin/bash
#SBATCH --job-name=anthropic-repl-alpha-sweep-l17
#SBATCH --output=/home/3242106/logs/alpha_sweep_l17_%j.out
#SBATCH --error=/home/3242106/logs/alpha_sweep_l17_%j.err
#SBATCH --time=23:59:00         # 9 traits × (1 baseline + 4 alphas × 100 gens) ≈ 7-8h LLM-judge + ~30 min logprob; cushion for judge rate-limit retries
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --qos=stud
#SBATCH --account=3242106
#SBATCH --partition=stud
#SBATCH --chdir=/home/3242106/steering-vector-composition-cloned
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=edoardo.panella@studbocconi.it

set -euo pipefail

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

mkdir -p /home/3242106/logs

echo "Starting alpha-sweep-l17 (unit-norm) — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/validation/run_alpha_sweep_l17.py
echo "alpha-sweep-l17 done — $(date)"
