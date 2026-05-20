# LoRA Fine-tuning Result

Date: 2026-05-19
GPU: NVIDIA H800 80GB
Base checkpoint: full_sft_h800_2h_768.pth
Data: lora_identity.jsonl
Job: mm_lora
LoRA weight: lora_identity_h800_768.pth

Final loss samples:
- epoch 9 step 5/6: 2.0665
- epoch 9 step 6/6: 2.2549
- epoch 10 step 1/6: 2.1569
- epoch 10 step 2/6: 2.1980
- epoch 10 step 3/6: 2.2567
- epoch 10 step 4/6: 2.1245
- epoch 10 step 5/6: 2.0973
- epoch 10 step 6/6: 2.1823

Saved weights:
- /home/hlihw/minimind/out/lora_identity_h800_768.pth
- /home/hlihw/minimind/checkpoints/lora_identity_h800_768.pth
- /home/hlihw/minimind/checkpoints/lora_identity_h800_768_resume.pth

Status: LoRA fine-tuning completed successfully.
