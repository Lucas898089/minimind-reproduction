#!/bin/bash
#SBATCH --job-name=stage_loss
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

date
hostname
nvidia-smi

python -s scripts/evaluate_stage_losses.py \
  --max_samples 512 \
  --batch_size 8 \
  --pretrain_max_len 340 \
  --sft_max_len 768 \
  --dpo_max_len 768 \
  --beta 0.15

date
