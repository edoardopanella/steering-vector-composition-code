#!/bin/bash
#SBATCH --job-name=human-eval-judge-l17
#SBATCH --output=/home/3242106/logs/human_eval_judge_%j.out
#SBATCH --error=/home/3242106/logs/human_eval_judge_%j.err
#SBATCH --time=04:00:00          # 3 pairs × 20 prompts = 60 completions (gen) + 60 × 3 judge calls
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

# NOTE: this run generates on GPU AND judges via OpenAI in one process, so the
# compute node MUST have outbound network (api.openai.com). HF Hub left online
# too; model is cached so load works either way.
export PYTHONPATH=/home/3242106/steering-vector-composition-cloned

echo "Starting human-eval generate + judge — $(date)"
python -u -m scripts.human_eval.human_evaluation
echo "human-eval generate + judge done — $(date)"

conda deactivate
module unload miniconda3
echo "End"
