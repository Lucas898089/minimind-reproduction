#!/bin/bash
#SBATCH --job-name=dpo_v2_ab
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
mkdir -p out checkpoints /home/hlihw/minimind_project/results/ablations/dpo_v2

date
hostname
nvidia-smi

run_dpo () {
  name=$1
  maxlen=$2
  beta=$3
  lr=$4

  echo "===== DPO ${name} START maxlen=${maxlen} beta=${beta} lr=${lr} ====="
  cd /home/hlihw/minimind/trainer

  python train_dpo.py \
    --epochs 1 \
    --batch_size 8 \
    --accumulation_steps 2 \
    --num_workers 4 \
    --log_interval 20 \
    --save_interval 10000 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len ${maxlen} \
    --data_path ../dataset/dpo.jsonl \
    --from_weight full_sft_h800_2h \
    --save_weight ${name} \
    --beta ${beta} \
    --learning_rate ${lr}

  echo "===== DPO ${name} DONE ====="
}

run_eval () {
  name=$1
  echo "===== EVAL ${name} ====="
  cd /home/hlihw/minimind

  printf "0\n" | python eval_llm.py \
    --load_from model \
    --save_dir out \
    --weight ${name} \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_new_tokens 128 \
    --temperature 0.7 \
    --top_p 0.9 \
    > /home/hlihw/minimind_project/results/ablations/dpo_v2/eval_${name}.out
}

run_dpo dpo_len1024_beta015_lr4e8 1024 0.15 4e-8
run_eval dpo_len1024_beta015_lr4e8

run_dpo dpo_len1024_beta005_lr4e8 1024 0.05 4e-8
run_eval dpo_len1024_beta005_lr4e8

run_dpo dpo_len1024_beta015_lr1e7 1024 0.15 1e-7
run_eval dpo_len1024_beta015_lr1e7

date
