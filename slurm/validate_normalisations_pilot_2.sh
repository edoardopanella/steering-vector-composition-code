#!/bin/bash
#SBATCH --job-name=pilot2-norm-modes
#SBATCH --output=/home/3247897/logs/validate_normalisations_pilot_2_%j.out
#SBATCH --error=/home/3247897/logs/validate_normalisations_pilot_2_%j.err
#SBATCH --time=05:00:00            # Pilot 2: 18 joint CSVs + 9 single α=3 CSVs = ~2,700 generations. Pro-rate from pilot 1 (~2.5h for 3,600) ≈ 1.5h + model-load + buffer.
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load miniconda3
source activate steering-vector-composition-venv

set -a; source .env; set +a

mkdir -p /home/3247897/logs

export COMPOSITION_PILOT_MODE=generate
export PYTHONPATH=/home/3247897/steering-vector-composition

echo "============================================================"
echo "Pilot 2 — generate stage — $(date)"
echo "Step 1/2: joint pilot 2 (18 new joint CSVs at α∈{5,6} normalize=True and α=3 normalize=per_axis)"
echo "============================================================"
python -u -m scripts.compositions.validate_normalisations_pilot_2

echo ""
echo "============================================================"
echo "Step 2/2: single-vector α=3 sweep (9 traits)"
echo "============================================================"
python -u -m scripts.validation.run_single_alpha3

echo ""
echo "Pilot 2 generate stage done — $(date)"
echo ""
echo "Next step (laptop, needs OPENAI_API_KEY):"
echo "  # joint pilot 2 — existing judge wrapper picks up new CSVs automatically"
echo "  python -m scripts.compositions.validate_normalisations_judge_local"
echo "  # then aggregate"
echo "  COMPOSITION_PILOT_MODE=judge python -m scripts.compositions.validate_normalisations_pilot_2"
echo "  # single α=3 — separate wrapper"
echo "  python -m scripts.validation.run_single_alpha3_judge_local"
