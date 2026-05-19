# Pretraining Result

Date: 2026-05-19
GPU: NVIDIA H800 80GB
Model: MiniMind dense, hidden_size=768, num_hidden_layers=8
Data: pretrain_t2t_mini.jsonl
Job: mm_pretrain-416138

Final loss samples:
- step 79260: 2.2576
- step 79280: 1.8989
- step 79300: 1.8103
- step 79320: 2.1842
- step 79340: 2.0598
- step 79360: 1.8947
- step 79380: 1.9389
- step 79390: 1.7927

Saved weights:
- /home/hlihw/minimind/out/pretrain_h800_2h_768.pth
- /home/hlihw/minimind/checkpoints/pretrain_h800_2h_768.pth
- /home/hlihw/minimind/checkpoints/pretrain_h800_2h_768_resume.pth

Status: Pretraining completed successfully.
