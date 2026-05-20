#!/bin/bash
#SBATCH --job-name=mm_dpo
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hlihw/minimind_project/results/logs/%x-%j.out
#SBATCH --error=/home/hlihw/minimind_project/results/logs/%x-%j.err

set -euo pipefail

module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hlihw/envs/minimind_py310
export PYTHONNOUSERSITE=1

cd /home/hlihw/minimind
mkdir -p out checkpoints /home/hlihw/minimind_project/results/logs

echo "===== JOB INFO ====="
date
hostname
nvidia-smi
python -V
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
PY

echo "===== DPO START ====="
cd trainer
python train_dpo.py \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --num_workers 4 \
  --log_interval 10 \
  --save_interval 500 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --data_path ../dataset/dpo.jsonl \
  --from_weight full_sft_h800_2h \
  --save_weight dpo_h800_2h

echo "===== DPO DONE ====="
date
