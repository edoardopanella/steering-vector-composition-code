#!/bin/bash

#SBATCH --account=3246955
#SBATCH --job-name=humev
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=/home/3246955/logs/%x_%j.out
#SBATCH --error=/home/3246955/logs/%x_%j.err
#SBATCH --chdir=/home/3246955/steering-vector-composition/

set -euo pipefail

module purge
module load miniconda3
source activate steer-vec

set -a; source /mnt/beegfsstudents/home/3246955/steering-vector-composition/.env; set +a

python -u -m scripts.human_evaluation

conda deactivate

module unload miniconda3
echo "End"
