# DPO Data Quality Profile

Profiled up to 20000 samples from `dpo.jsonl`.

Key checks:
- schema validity
- empty prompt/chosen/rejected
- chosen/rejected length distribution
- prompt mismatch between chosen and rejected
- answer similarity
- think-tag residue
- unclosed code fences
- rough repetition ratio

See `dpo_quality_profile.csv` and `dpo_suspicious_examples.jsonl`.
