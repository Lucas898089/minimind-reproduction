import csv
import re
from pathlib import Path

root = Path.home() / "minimind_project"
logs = root / "results/logs"
out = root / "results/tables"
out.mkdir(parents=True, exist_ok=True)

runs = [
    {
        "stage": "pretrain",
        "job_id": "416138",
        "log": "mm_pretrain-416138.out",
        "data": "pretrain_t2t_mini.jsonl",
        "base_weight": "random_init",
        "output_weight": "pretrain_h800_2h_768.pth",
        "hidden_size": 768,
        "layers": 8,
        "batch_size": 16,
        "accumulation_steps": 2,
        "max_seq_len": 340,
    },
    {
        "stage": "sft",
        "job_id": "416159",
        "log": "mm_sft-416159.out",
        "data": "sft_t2t_mini.jsonl",
        "base_weight": "pretrain_h800_2h_768.pth",
        "output_weight": "full_sft_h800_2h_768.pth",
        "hidden_size": 768,
        "layers": 8,
        "batch_size": 16,
        "accumulation_steps": 2,
        "max_seq_len": 768,
    },
    {
        "stage": "lora",
        "job_id": "416367",
        "log": "mm_lora-416367.out",
        "data": "lora_identity.jsonl",
        "base_weight": "full_sft_h800_2h_768.pth",
        "output_weight": "lora_identity_h800_768.pth",
        "hidden_size": 768,
        "layers": 8,
        "batch_size": 16,
        "accumulation_steps": 1,
        "max_seq_len": 512,
    },
    {
        "stage": "dpo",
        "job_id": "416570",
        "log": "mm_dpo-416570.out",
        "data": "dpo.jsonl",
        "base_weight": "full_sft_h800_2h_768.pth",
        "output_weight": "dpo_h800_2h_768.pth",
        "hidden_size": 768,
        "layers": 8,
        "batch_size": 8,
        "accumulation_steps": 2,
        "max_seq_len": 768,
    },
]

loss_patterns = [
    re.compile(r"Epoch:\[\d+/\d+\]\((\d+)/(\d+)\), loss: ([0-9.]+)"),
]

def parse_log(path):
    text = path.read_text(errors="ignore") if path.exists() else ""
    lines = text.splitlines()
    final_step = final_total = final_loss = ""
    for line in lines:
        for p in loss_patterns:
            m = p.search(line)
            if m:
                final_step, final_total, final_loss = m.group(1), m.group(2), m.group(3)

    hostname = ""
    gpu = ""
    start_time = ""
    end_time = ""

    for i, line in enumerate(lines):
        if "===== JOB INFO =====" in line and i + 1 < len(lines):
            start_time = lines[i + 1]
        if line.startswith("dgx-"):
            hostname = line.strip()
        if "NVIDIA H800" in line:
            gpu = "NVIDIA H800 80GB"
        if " DONE =====" in line and i + 1 < len(lines):
            end_time = lines[i + 1]

    return {
        "start_time": start_time,
        "end_time": end_time,
        "hostname": hostname,
        "gpu": gpu,
        "final_step": final_step,
        "total_steps": final_total,
        "final_train_loss": final_loss,
    }

rows = []
for r in runs:
    meta = parse_log(logs / r["log"])
    row = {**r, **meta}
    row["effective_batch"] = int(r["batch_size"]) * int(r["accumulation_steps"])
    rows.append(row)

fields = [
    "stage", "job_id", "log", "data", "base_weight", "output_weight",
    "hidden_size", "layers", "batch_size", "accumulation_steps", "effective_batch",
    "max_seq_len", "final_step", "total_steps", "final_train_loss",
    "start_time", "end_time", "hostname", "gpu"
]

with (out / "experiment_log.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(out / "experiment_log.csv")
