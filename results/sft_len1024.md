# SFT Len1024 Ablation

Date: 2026-06-05
GPU: NVIDIA H800 80GB
Base checkpoint: pretrain_h800_2h_768.pth
Data: sft_t2t_mini.jsonl
Change: max_seq_len 768 -> 1024
Weight: full_sft_len1024_h800_2h

## Training Result

Final training log sample:

- step 113080/113215: loss 1.2752
- step 113100/113215: loss 1.8695
- step 113120/113215: loss 1.8793
- step 113140/113215: loss 1.5426
- step 113160/113215: loss 1.3856
- step 113180/113215: loss 1.6778
- step 113200/113215: loss 1.8268
- step 113215/113215: loss 1.7008

## Diagnostic Comparison

On the same stage-loss diagnostic subset:

| model | sft assistant CE | ppl | dpo pref loss | chosen acc |
|---|---:|---:|---:|---:|
| SFT len768 | 1.364137 | 3.912345 | 0.693147 | 0.605469 |
| SFT len1024 | 1.364257 | 3.912814 | 0.711511 | 0.599609 |

## Interpretation

Increasing SFT max_seq_len from 768 to 1024 did not improve assistant CE or qualitative generation. The current bottleneck is likely not context truncation alone, but SFT data quality, task coverage, encoding pollution, and model capacity.
