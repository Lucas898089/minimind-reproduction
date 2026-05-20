#!/bin/bash
#SBATCH --job-name=mm_lora
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

echo "===== LORA START ====="
cd trainer
python train_lora.py \
  --epochs 10 \
  --batch_size 16 \
  --accumulation_steps 1 \
  --num_workers 4 \
  --log_interval 1 \
  --save_interval 20 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 512 \
  --data_path ../dataset/lora_identity.jsonl \
  --from_weight full_sft_h800_2h \
  --lora_name lora_identity_h800

echo "===== LORA DONE ====="
date
