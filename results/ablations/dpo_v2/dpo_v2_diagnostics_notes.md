# DPO v2 Diagnostics

We evaluated SFT, the previous DPO checkpoint, and three new DPO ablations on the same diagnostic subset.

## Key Result

DPO variants did not significantly improve preference diagnostics over the SFT baseline.

| model | sft assistant loss | dpo pref loss | chosen accuracy |
|---|---:|---:|---:|
| SFT | 1.364644 | 0.693147 | 0.605469 |
| old DPO len768 beta0.15 lr4e-8 | 1.364754 | 0.695710 | 0.609375 |
| DPO len1024 beta0.15 lr4e-8 | 1.365763 | 0.697518 | 0.605469 |
| DPO len1024 beta0.05 lr4e-8 | 1.364897 | 0.697266 | 0.605469 |
| DPO len1024 beta0.15 lr1e-7 | 1.364289 | 0.691827 | 0.607422 |

## Interpretation

Increasing DPO max_seq_len from 768 to 1024 did not produce a clear improvement. Beta and learning-rate changes within the tested range also had minimal effect.

This suggests that the current bottleneck is likely not just DPO hyperparameters. The next step should focus on preference-data quality, reward signal strength, and whether chosen/rejected pairs are sufficiently distinguishable for a 64M model.
