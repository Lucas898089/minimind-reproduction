import csv
import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer

MINIMIND_ROOT = Path("/home/hlihw/minimind")
DATA_DIR = MINIMIND_ROOT / "dataset"
OUT_DIR = Path.home() / "minimind_project/results/data_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(str(MINIMIND_ROOT / "model"))


def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * p / 100)))
    return values[idx]


def summarize(name, lengths, limits):
    row = {
        "dataset": name,
        "count": len(lengths),
        "p50": pct(lengths, 50),
        "p90": pct(lengths, 90),
        "p95": pct(lengths, 95),
        "p99": pct(lengths, 99),
        "max": max(lengths) if lengths else 0,
        "mean": round(statistics.mean(lengths), 2) if lengths else 0,
    }
    for lim in limits:
        row[f"truncate_ratio_gt_{lim}"] = round(sum(x > lim for x in lengths) / max(len(lengths), 1), 4)
    return row


def iter_jsonl(path, max_samples=None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                yield json.loads(line)


rows = []

# Pretrain: plain text length
pre_lengths = []
for obj in iter_jsonl(DATA_DIR / "pretrain_t2t_mini.jsonl"):
    ids = tok(str(obj["text"]), add_special_tokens=False).input_ids
    pre_lengths.append(len(ids) + 2)  # BOS/EOS
rows.append(summarize("pretrain_t2t_mini", pre_lengths, [256, 340, 512, 768, 1024]))

# SFT: chat template total length and assistant label length approximation.
sft_total = []
sft_assistant_chars = []
for obj in iter_jsonl(DATA_DIR / "sft_t2t_mini.jsonl"):
    conv = obj["conversations"]
    prompt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    sft_total.append(len(tok(prompt).input_ids))
    assistant_text = "\n".join(m.get("content", "") for m in conv if m.get("role") == "assistant")
    sft_assistant_chars.append(len(assistant_text))
rows.append(summarize("sft_t2t_mini_total", sft_total, [512, 768, 1024, 1536, 2048]))
rows.append(summarize("sft_t2t_mini_assistant_chars", sft_assistant_chars, [256, 512, 768, 1024, 1536]))

# LoRA identity
lora_total = []
for obj in iter_jsonl(DATA_DIR / "lora_identity.jsonl"):
    prompt = tok.apply_chat_template(obj["conversations"], tokenize=False, add_generation_prompt=False)
    lora_total.append(len(tok(prompt).input_ids))
rows.append(summarize("lora_identity_total", lora_total, [256, 512, 768, 1024]))

# DPO chosen/rejected
chosen_lengths = []
rejected_lengths = []
for obj in iter_jsonl(DATA_DIR / "dpo.jsonl"):
    c = tok.apply_chat_template(obj["chosen"], tokenize=False, add_generation_prompt=False)
    r = tok.apply_chat_template(obj["rejected"], tokenize=False, add_generation_prompt=False)
    chosen_lengths.append(len(tok(c).input_ids))
    rejected_lengths.append(len(tok(r).input_ids))
rows.append(summarize("dpo_chosen_total", chosen_lengths, [512, 768, 1024, 2048, 4096]))
rows.append(summarize("dpo_rejected_total", rejected_lengths, [512, 768, 1024, 2048, 4096]))

csv_path = OUT_DIR / "data_length_profile.csv"
fields = sorted({k for row in rows for k in row.keys()})
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

md_path = OUT_DIR / "data_profile_notes.md"
md_path.write_text(
    "# MiniMind Data Profile\n\n"
    "This profile summarizes token-length distributions and truncation ratios for the datasets used in the reproduction.\n\n"
    "- Pretrain lengths include BOS/EOS.\n"
    "- SFT lengths are based on tokenizer chat_template output.\n"
    "- DPO lengths are computed separately for chosen and rejected conversations.\n",
    encoding="utf-8"
)

print(csv_path)
print(md_path)
for row in rows:
    print(row)
