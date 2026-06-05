#!/bin/bash
#SBATCH --job-name=mm_sft1024
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hlihw/minimind_project/results/logs/%x-%j.out
#SBATCH --error=/home/hlihw/minimind_project/results/logs/%x-%j.err

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

echo "===== FULL SFT LEN1024 START ====="
cd trainer
python train_full_sft.py \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --num_workers 8 \
  --log_interval 20 \
  --save_interval 1000 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 1024 \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain_h800_2h \
  --save_weight full_sft_len1024_h800_2h

echo "===== FULL SFT LEN1024 DONE ====="
date
