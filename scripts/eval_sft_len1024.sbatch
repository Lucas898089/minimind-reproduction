#!/bin/bash
#SBATCH --job-name=eval_sft1024
#SBATCH --account=mscbdtsuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/hlihw/minimind_project/results/logs/%x-%j.out
#SBATCH --error=/home/hlihw/minimind_project/results/logs/%x-%j.err

module load Anaconda3/2023.09-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hlihw/envs/minimind_py310
export PYTHONNOUSERSITE=1

cd /home/hlihw/minimind

printf "0\n" | python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_len1024_h800_2h \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.9
