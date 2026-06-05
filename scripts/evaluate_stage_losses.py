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
from model.model_lora import apply_lora, load_lora
from dataset.lm_dataset import PretrainDataset, SFTDataset, DPODataset
from trainer.trainer_utils import init_model


MODELS = {
    "pretrain": {"weight": "pretrain_h800_2h", "lora": None},
    "sft": {"weight": "full_sft_h800_2h", "lora": None},
    "lora": {"weight": "full_sft_h800_2h", "lora": "lora_identity_h800"},
    "dpo": {"weight": "dpo_h800_2h", "lora": None},
}


class FirstN(Dataset):
    def __init__(self, ds, n):
        self.ds = ds
        self.n = min(len(ds), n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.ds[idx]


def load_eval_model(name, hidden_size, layers, device):
    spec = MODELS[name]
    cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=layers)
    model, tokenizer = init_model(
        cfg,
        spec["weight"],
        tokenizer_path=str(MINIMIND_ROOT / "model"),
        save_dir=str(MINIMIND_ROOT / "out"),
        device=device,
    )

    if spec["lora"]:
        apply_lora(model)
        lora_path = MINIMIND_ROOT / "out" / f"{spec['lora']}_{hidden_size}.pth"
        load_lora(model, str(lora_path))

    model.eval()
    return model, tokenizer


@torch.no_grad()
def eval_ce_loss(model, loader, device, dtype):
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(dtype=dtype):
            out = model(input_ids, labels=labels)
            loss = out.loss

        valid_tokens = (labels != -100).sum().item()
        total_loss += float(loss.item()) * valid_tokens
        total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl, total_tokens


def sequence_logps(logits, labels, mask):
    safe_labels = labels.masked_fill(labels < 0, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=1)


@torch.no_grad()
def eval_dpo_preference(model, ref_model, loader, device, dtype, beta):
    total = 0
    correct = 0
    total_policy_margin = 0.0
    total_ref_margin = 0.0
    total_dpo_loss = 0.0

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
            policy_logits = model(x).logits
            ref_logits = ref_model(x).logits

            policy_logps = sequence_logps(policy_logits, y, mask)
            ref_logps = sequence_logps(ref_logits, y, mask)

            bsz = x_chosen.shape[0]
            p_chosen = policy_logps[:bsz]
            p_rejected = policy_logps[bsz:]
            r_chosen = ref_logps[:bsz]
            r_rejected = ref_logps[bsz:]

            policy_margin = p_chosen - p_rejected
            ref_margin = r_chosen - r_rejected
            dpo_logits = policy_margin - ref_margin
            dpo_loss = -F.logsigmoid(beta * dpo_logits)

        total += bsz
        correct += (policy_margin > 0).sum().item()
        total_policy_margin += policy_margin.sum().item()
        total_ref_margin += ref_margin.sum().item()
        total_dpo_loss += dpo_loss.sum().item()

    return {
        "loss": total_dpo_loss / max(total, 1),
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
    parser.add_argument("--pretrain_max_len", type=int, default=340)
    parser.add_argument("--sft_max_len", type=int, default=768)
    parser.add_argument("--dpo_max_len", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--out_dir", default="/home/hlihw/minimind_project/results/diagnostics")
    args = parser.parse_args()

    device = "cuda:0"
    dtype = torch.bfloat16
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer from SFT model...", flush=True)
    _, tokenizer = load_eval_model("sft", args.hidden_size, args.layers, device)

    pretrain_ds = FirstN(
        PretrainDataset(str(MINIMIND_ROOT / "dataset/pretrain_t2t_mini.jsonl"), tokenizer, max_length=args.pretrain_max_len),
        args.max_samples,
    )
    sft_ds = FirstN(
        SFTDataset(str(MINIMIND_ROOT / "dataset/sft_t2t_mini.jsonl"), tokenizer, max_length=args.sft_max_len),
        args.max_samples,
    )
    dpo_ds = FirstN(
        DPODataset(str(MINIMIND_ROOT / "dataset/dpo.jsonl"), tokenizer, max_length=args.dpo_max_len),
        args.max_samples,
    )

    pretrain_loader = DataLoader(pretrain_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    sft_loader = DataLoader(sft_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dpo_loader = DataLoader(dpo_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rows = []

    for model_name in ["pretrain", "sft", "lora", "dpo"]:
        print(f"===== CE diagnostics: {model_name} =====", flush=True)
        model, _ = load_eval_model(model_name, args.hidden_size, args.layers, device)

        pre_loss, pre_ppl, pre_tokens = eval_ce_loss(model, pretrain_loader, device, dtype)
        sft_loss, sft_ppl, sft_tokens = eval_ce_loss(model, sft_loader, device, dtype)

        rows.append({
            "model": model_name,
            "metric_group": "pretrain_ce",
            "loss": round(pre_loss, 6),
            "ppl": round(pre_ppl, 6),
            "tokens_or_pairs": pre_tokens,
            "chosen_accuracy": "",
            "avg_policy_margin": "",
            "avg_ref_margin": "",
        })
        rows.append({
            "model": model_name,
            "metric_group": "sft_assistant_ce",
            "loss": round(sft_loss, 6),
            "ppl": round(sft_ppl, 6),
            "tokens_or_pairs": sft_tokens,
            "chosen_accuracy": "",
            "avg_policy_margin": "",
            "avg_ref_margin": "",
        })

        print(f"{model_name} pretrain_ce loss={pre_loss:.4f} ppl={pre_ppl:.2f}", flush=True)
        print(f"{model_name} sft_assistant_ce loss={sft_loss:.4f} ppl={sft_ppl:.2f}", flush=True)

        del model
        torch.cuda.empty_cache()

    print("===== Loading SFT reference for DPO diagnostics =====", flush=True)
    ref_model, _ = load_eval_model("sft", args.hidden_size, args.layers, device)
    ref_model.eval()

    for model_name in ["sft", "lora", "dpo"]:
        print(f"===== DPO diagnostics: {model_name} =====", flush=True)
        model, _ = load_eval_model(model_name, args.hidden_size, args.layers, device)
        result = eval_dpo_preference(model, ref_model, dpo_loader, device, dtype, args.beta)

        rows.append({
            "model": model_name,
            "metric_group": "dpo_preference",
            "loss": round(result["loss"], 6),
            "ppl": "",
            "tokens_or_pairs": result["pairs"],
            "chosen_accuracy": round(result["chosen_accuracy"], 6),
            "avg_policy_margin": round(result["avg_policy_margin"], 6),
            "avg_ref_margin": round(result["avg_ref_margin"], 6),
        })

        print(model_name, result, flush=True)
        del model
        torch.cuda.empty_cache()

    csv_path = out_dir / "stage_loss_diagnostics.csv"
    fields = [
        "model", "metric_group", "loss", "ppl", "tokens_or_pairs",
        "chosen_accuracy", "avg_policy_margin", "avg_ref_margin"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("saved", csv_path, flush=True)


if __name__ == "__main__":
    main()
