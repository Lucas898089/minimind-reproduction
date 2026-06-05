import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MINIMIND_ROOT = Path("/home/hlihw/minimind")
sys.path.insert(0, str(MINIMIND_ROOT))

from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset, DPODataset
from trainer.trainer_utils import init_model


MODELS = {
    "sft": "full_sft_h800_2h",
    "dpo_old_len768_beta015_lr4e8": "dpo_h800_2h",
    "dpo_len1024_beta015_lr4e8": "dpo_len1024_beta015_lr4e8",
    "dpo_len1024_beta005_lr4e8": "dpo_len1024_beta005_lr4e8",
    "dpo_len1024_beta015_lr1e7": "dpo_len1024_beta015_lr1e7",
}


class FirstN(Dataset):
    def __init__(self, ds, n):
        self.ds = ds
        self.n = min(len(ds), n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.ds[idx]


def load_model(weight, hidden_size, layers, device):
    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=layers)
    model, tokenizer = init_model(
        cfg,
        weight,
        tokenizer_path=str(MINIMIND_ROOT / "model"),
        save_dir=str(MINIMIND_ROOT / "out"),
        device=device,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def eval_ce_loss(model, loader, device, dtype):
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.cuda.amp.autocast(dtype=dtype):
            out = model(input_ids, labels=labels)
            loss = out.loss
        valid = (labels != -100).sum().item()
        total_loss += loss.item() * valid
        total_tokens += valid

    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg) if avg < 50 else float("inf")
    return avg, ppl, total_tokens


def seq_logps(logits, labels, mask):
    safe_labels = labels.masked_fill(labels < 0, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=1)


@torch.no_grad()
def eval_pref(model, ref_model, loader, device, dtype, beta):
    total = 0
    correct = 0
    total_policy_margin = 0.0
    total_ref_margin = 0.0
    total_loss = 0.0

    for batch in loader:
        x_chosen = batch["x_chosen"].to(device)
        x_rejected = batch["x_rejected"].to(device)
        y_chosen = batch["y_chosen"].to(device)
        y_rejected = batch["y_rejected"].to(device)
        mask_chosen = batch["mask_chosen"].to(device)
        mask_rejected = batch["mask_rejected"].to(device)

        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        with torch.cuda.amp.autocast(dtype=dtype):
            p_logits = model(x).logits
            r_logits = ref_model(x).logits

            p_logps = seq_logps(p_logits, y, mask)
            r_logps = seq_logps(r_logits, y, mask)

            bsz = x_chosen.shape[0]
            p_margin = p_logps[:bsz] - p_logps[bsz:]
            r_margin = r_logps[:bsz] - r_logps[bsz:]
            dpo_logits = p_margin - r_margin
            loss = -F.logsigmoid(beta * dpo_logits)

        total += bsz
        correct += (p_margin > 0).sum().item()
        total_policy_margin += p_margin.sum().item()
        total_ref_margin += r_margin.sum().item()
        total_loss += loss.sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "chosen_accuracy": correct / max(total, 1),
        "avg_policy_margin": total_policy_margin / max(total, 1),
        "avg_ref_margin": total_ref_margin / max(total, 1),
        "pairs": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sft_max_len", type=int, default=1024)
    parser.add_argument("--dpo_max_len", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--out_dir", default="/home/hlihw/minimind_project/results/diagnostics")
    args = parser.parse_args()

    device = "cuda:0"
    dtype = torch.bfloat16
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_model, tokenizer = load_model("full_sft_h800_2h", args.hidden_size, args.layers, device)

    sft_ds = FirstN(
        SFTDataset(str(MINIMIND_ROOT / "dataset/sft_t2t_mini.jsonl"), tokenizer, max_length=args.sft_max_len),
        args.max_samples,
    )
    dpo_ds = FirstN(
        DPODataset(str(MINIMIND_ROOT / "dataset/dpo.jsonl"), tokenizer, max_length=args.dpo_max_len),
        args.max_samples,
    )
    sft_loader = DataLoader(sft_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dpo_loader = DataLoader(dpo_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rows = []
    for name, weight in MODELS.items():
        print(f"===== {name} =====", flush=True)
        model, _ = load_model(weight, args.hidden_size, args.layers, device)

        sft_loss, sft_ppl, sft_tokens = eval_ce_loss(model, sft_loader, device, dtype)
        pref = eval_pref(model, ref_model, dpo_loader, device, dtype, args.beta)

        row = {
            "model": name,
            "weight": weight,
            "sft_assistant_loss": round(sft_loss, 6),
            "sft_assistant_ppl": round(sft_ppl, 6),
            "sft_tokens": sft_tokens,
            "dpo_pref_loss": round(pref["loss"], 6),
            "chosen_accuracy": round(pref["chosen_accuracy"], 6),
            "avg_policy_margin": round(pref["avg_policy_margin"], 6),
            "avg_ref_margin": round(pref["avg_ref_margin"], 6),
            "pairs": pref["pairs"],
        }
        rows.append(row)
        print(row, flush=True)

        del model
        torch.cuda.empty_cache()

    csv_path = out_dir / "dpo_v2_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("saved", csv_path)


if __name__ == "__main__":
    main()
