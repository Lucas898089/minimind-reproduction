import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

MINIMIND_ROOT = Path("/home/hlihw/minimind")
sys.path.insert(0, str(MINIMIND_ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from transformers import AutoTokenizer


MODEL_SPECS = {
    "pretrain": {"weight": "pretrain_h800_2h", "lora": None},
    "sft": {"weight": "full_sft_h800_2h", "lora": None},
    "lora": {"weight": "full_sft_h800_2h", "lora": "lora_identity_h800"},
    "dpo": {"weight": "dpo_h800_2h", "lora": None},
}

TNEWS_LABELS = [
    ("0", "故事"),
    ("1", "文化"),
    ("2", "娱乐"),
    ("3", "体育"),
    ("4", "财经"),
    ("5", "房产"),
    ("6", "汽车"),
    ("7", "教育"),
    ("8", "科技"),
    ("9", "军事"),
    ("10", "旅游"),
    ("11", "国际"),
    ("12", "股票"),
    ("13", "农业"),
    ("14", "游戏"),
]

OCNLI_LABELS = [
    ("entailment", "蕴含"),
    ("neutral", "中立"),
    ("contradiction", "矛盾"),
]


def load_clue(task, split):
    for repo in ["clue", "clue/clue"]:
        try:
            return load_dataset(repo, task, split=split, trust_remote_code=True)
        except Exception:
            pass
    raise RuntimeError(f"Cannot load CLUE task: {task}")


def init_model(model_name, hidden_size, num_layers):
    spec = MODEL_SPECS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(str(MINIMIND_ROOT / "model"))

    model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers)
    )

    weight_path = MINIMIND_ROOT / "out" / f"{spec['weight']}_{hidden_size}.pth"
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    if spec["lora"]:
        apply_lora(model)
        lora_path = MINIMIND_ROOT / "out" / f"{spec['lora']}_{hidden_size}.pth"
        load_lora(model, str(lora_path))

    model = model.half().eval().cuda()
    return model, tokenizer


def build_prompt(task, ex):
    if task == "afqmc":
        prompt = (
            "判断下面两个中文句子的语义是否相同。\n"
            f"句子1：{ex['sentence1']}\n"
            f"句子2：{ex['sentence2']}\n"
            "A. 不相同\n"
            "B. 相同\n"
            "答案："
        )
        choices = [("0", "A"), ("1", "B")]
        gold = str(ex["label"])
        return prompt, choices, gold

    if task == "tnews":
        prompt = "判断下面新闻标题属于哪个类别。\n"
        prompt += f"标题：{ex['sentence']}\n"
        letters = "ABCDEFGHIJKLMNO"
        choices = []
        for i, (label, name) in enumerate(TNEWS_LABELS):
            prompt += f"{letters[i]}. {name}\n"
            choices.append((label, letters[i]))
        prompt += "答案："
        gold = str(ex["label"])
        return prompt, choices, gold

    if task == "ocnli":
        prompt = (
            "判断前提和假设之间的关系。\n"
            f"前提：{ex['sentence1']}\n"
            f"假设：{ex['sentence2']}\n"
            "A. 蕴含\n"
            "B. 中立\n"
            "C. 矛盾\n"
            "答案："
        )
        raw = ex["label"]
        gold = str(raw)
        if gold in {"0", "0.0"}:
            gold = "entailment"
        elif gold in {"1", "1.0"}:
            gold = "neutral"
        elif gold in {"2", "2.0"}:
            gold = "contradiction"
        choices = [("entailment", "A"), ("neutral", "B"), ("contradiction", "C")]
        return prompt, choices, gold

    raise ValueError(task)


def format_input(tokenizer, model_name, prompt):
    if model_name == "pretrain":
        return tokenizer.bos_token + prompt

    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, open_thinking=False
    )


@torch.no_grad()
def score_choice(model, tokenizer, prefix, choice):
    prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    choice_ids = tokenizer(choice, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    input_ids = torch.cat([prefix_ids, choice_ids], dim=1)
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)

    start = prefix_ids.shape[1]
    total = 0.0
    for i in range(choice_ids.shape[1]):
        token_id = choice_ids[0, i]
        total += log_probs[0, start + i - 1, token_id].item()

    return total / max(1, choice_ids.shape[1])


def evaluate_one(task, model_name, max_samples, split, hidden_size, num_layers, out_dir):
    ds = load_clue(task, split)
    model, tokenizer = init_model(model_name, hidden_size, num_layers)

    correct = 0
    total = 0
    rows = []

    for ex in tqdm(ds.select(range(min(max_samples, len(ds)))), desc=f"{task}-{model_name}"):
        if "label" not in ex or ex["label"] in [-1, None, ""]:
            continue

        prompt, choices, gold = build_prompt(task, ex)
        prefix = format_input(tokenizer, model_name, prompt)

        scores = []
        for label, letter in choices:
            scores.append((label, letter, score_choice(model, tokenizer, prefix, letter)))

        pred_label, pred_letter, pred_score = max(scores, key=lambda x: x[2])
        ok = str(pred_label) == str(gold)

        correct += int(ok)
        total += 1
        rows.append({
            "task": task,
            "model": model_name,
            "gold": str(gold),
            "pred": str(pred_label),
            "pred_letter": pred_letter,
            "correct": ok,
            "scores": scores,
        })

    acc = correct / total if total else math.nan
    out = {
        "task": task,
        "model": model_name,
        "split": split,
        "max_samples": max_samples,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "rows": rows[:20],
    }

    out_path = out_dir / f"{task}_{model_name}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    del model
    torch.cuda.empty_cache()
    return {"task": task, "model": model_name, "total": total, "correct": correct, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["afqmc", "tnews", "ocnli"])
    parser.add_argument("--models", nargs="+", default=["pretrain", "sft", "lora", "dpo"])
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--out_dir", default="/home/hlihw/minimind_project/results/benchmarks")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for task in args.tasks:
        for model_name in args.models:
            summary.append(
                evaluate_one(
                    task, model_name, args.max_samples, args.split,
                    args.hidden_size, args.num_hidden_layers, out_dir
                )
            )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "model", "total", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(summary)

    print("Saved:", csv_path)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
