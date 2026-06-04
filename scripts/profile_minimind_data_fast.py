import csv
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

MINIMIND_ROOT = Path("/home/hlihw/minimind")
DATA_DIR = MINIMIND_ROOT / "dataset"
OUT_DIR = Path.home() / "minimind_project/results/data_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 5000
tok = AutoTokenizer.from_pretrained(str(MINIMIND_ROOT / "model"))


def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int((len(values) - 1) * p / 100))
    return values[idx]


def summarize(name, lengths, limits):
    row = {
        "dataset": name,
        "sample_count": len(lengths),
        "mean": round(statistics.mean(lengths), 2) if lengths else 0,
        "p50": pct(lengths, 50),
        "p90": pct(lengths, 90),
        "p95": pct(lengths, 95),
        "p99": pct(lengths, 99),
        "max": max(lengths) if lengths else 0,
    }
    for lim in limits:
        row[f"ratio_gt_{lim}"] = round(sum(x > lim for x in lengths) / max(len(lengths), 1), 4)
    return row


def iter_jsonl(path, max_samples=MAX_SAMPLES):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if line.strip():
                yield json.loads(line)


rows = []

pre_lengths = []
for obj in iter_jsonl(DATA_DIR / "pretrain_t2t_mini.jsonl"):
    ids = tok(str(obj["text"]), add_special_tokens=False).input_ids
    pre_lengths.append(len(ids) + 2)
rows.append(summarize("pretrain_t2t_mini", pre_lengths, [256, 340, 512, 768, 1024]))

sft_total = []
sft_assistant_chars = []
sft_has_think = 0
for obj in iter_jsonl(DATA_DIR / "sft_t2t_mini.jsonl"):
    conv = obj["conversations"]
    prompt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    sft_total.append(len(tok(prompt).input_ids))
    assistant_text = "\n".join(m.get("content", "") or "" for m in conv if m.get("role") == "assistant")
    sft_assistant_chars.append(len(assistant_text))
    if "<think" in prompt or "</think" in prompt:
        sft_has_think += 1
rows.append(summarize("sft_t2t_mini_total", sft_total, [512, 768, 1024, 1536, 2048]))
rows.append(summarize("sft_t2t_mini_assistant_chars", sft_assistant_chars, [256, 512, 768, 1024, 1536]))

lora_total = []
for obj in iter_jsonl(DATA_DIR / "lora_identity.jsonl", max_samples=MAX_SAMPLES):
    prompt = tok.apply_chat_template(obj["conversations"], tokenize=False, add_generation_prompt=False)
    lora_total.append(len(tok(prompt).input_ids))
rows.append(summarize("lora_identity_total", lora_total, [256, 512, 768, 1024]))

chosen_lengths = []
rejected_lengths = []
for obj in iter_jsonl(DATA_DIR / "dpo.jsonl"):
    c = tok.apply_chat_template(obj["chosen"], tokenize=False, add_generation_prompt=False)
    r = tok.apply_chat_template(obj["rejected"], tokenize=False, add_generation_prompt=False)
    chosen_lengths.append(len(tok(c).input_ids))
    rejected_lengths.append(len(tok(r).input_ids))
rows.append(summarize("dpo_chosen_total", chosen_lengths, [512, 768, 1024, 2048, 4096]))
rows.append(summarize("dpo_rejected_total", rejected_lengths, [512, 768, 1024, 2048, 4096]))

csv_path = OUT_DIR / "data_length_profile_fast.csv"
fields = sorted({k for row in rows for k in row.keys()})
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

notes = OUT_DIR / "data_profile_notes.md"
notes.write_text(
    f"# MiniMind Data Profile\n\n"
    f"Fast profile based on up to {MAX_SAMPLES} samples per dataset.\n\n"
    f"SFT samples containing think tags in sampled subset: {sft_has_think}/{len(sft_total)}.\n\n"
    "Use this profile to decide max_seq_len and understand truncation ratios.\n",
    encoding="utf-8"
)

print(csv_path)
for row in rows:
    print(row)
