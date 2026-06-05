#!/bin/bash
#SBATCH --job-name=clue_eval
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/home/hlihw/minimind_project/results/logs/%x-%j.out
#SBATCH --error=/home/hlihw/minimind_project/results/logs/%x-%j.err

set -euo pipefail

module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hlihw/envs/minimind_py310
export PYTHONNOUSERSITE=1

cd /home/hlihw/minimind_project

python scripts/evaluate_clue_choice.py \
  --tasks afqmc tnews ocnli \
  --models pretrain sft lora dpo \
  --max_samples 200 \
  --split validation
