#!/bin/bash
#SBATCH --job-name=anthropic-repl-layer-selection
#SBATCH --output=/home/3242106/logs/layer_selection_%j.out
#SBATCH --error=/home/3242106/logs/layer_selection_%j.err
#SBATCH --time=23:59:00         # 9 traits × (1 baseline + 32 layers) × ~20 gens; ~1.5h/trait + judge headroom
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

echo "Starting anthropic-repl layer-selection-all — $(date)"
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned
python -u scripts/layer_selection/run_layer_selection_all.py
echo "Anthropic-repl layer-selection-all done — $(date)"
