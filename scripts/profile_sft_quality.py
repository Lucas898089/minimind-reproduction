import csv, json, os, re
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer

ROOT = Path("/home/hlihw/minimind")
DATA = ROOT / "dataset/sft_t2t_mini.jsonl"
OUT = Path("/home/hlihw/minimind_project/results/data_profile")
OUT.mkdir(parents=True, exist_ok=True)

N = int(os.environ.get("SFT_MAX_SAMPLES", "20000"))
tok = AutoTokenizer.from_pretrained(str(ROOT / "model"), trust_remote_code=True)

def zh_ratio(s):
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    return sum("\u4e00" <= c <= "\u9fff" for c in chars) / len(chars)

def rep_ratio(s, n=4):
    s = re.sub(r"\s+", "", s)
    if len(s) < n * 4:
        return 0.0
    grams = [s[i:i+n] for i in range(len(s)-n+1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)

def token_len(convs):
    try:
        text = tok.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = "\n".join(str(x) for x in convs)
    return len(tok(text, add_special_tokens=False).input_ids)

def q(xs, p):
    if not xs:
        return ""
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)]

stats = {
    "sample_count": 0,
    "bad_schema": 0,
    "empty_user": 0,
    "empty_assistant": 0,
    "has_think_tag": 0,
    "mostly_non_zh": 0,
    "repeat_gt030": 0,
    "unclosed_code": 0,
    "total_tokens_gt768": 0,
    "total_tokens_gt1024": 0,
}

lens, reps, zhs = [], [], []

with DATA.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break
        if not line.strip():
            continue

        stats["sample_count"] += 1
        obj = json.loads(line)
        convs = obj.get("conversations")
        if not isinstance(convs, list):
            stats["bad_schema"] += 1
            continue

        users = [str(m.get("content") or "") for m in convs if m.get("role") == "user"]
        assistants = [str(m.get("content") or "") for m in convs if m.get("role") == "assistant"]
        all_text = "\n".join(str(m.get("content") or "") for m in convs)
        ans = "\n".join(assistants)

        if not "".join(users).strip():
            stats["empty_user"] += 1
        if not ans.strip():
            stats["empty_assistant"] += 1
        if "<think>" in all_text or "</think>" in all_text:
            stats["has_think_tag"] += 1
        if all_text.count("```") % 2 == 1:
            stats["unclosed_code"] += 1

        zr = zh_ratio(all_text)
        rr = rep_ratio(ans)
        tl = token_len(convs)

        lens.append(tl)
        reps.append(rr)
        zhs.append(zr)

        if zr < 0.35:
            stats["mostly_non_zh"] += 1
        if rr > 0.30:
            stats["repeat_gt030"] += 1
        if tl > 768:
            stats["total_tokens_gt768"] += 1
        if tl > 1024:
            stats["total_tokens_gt1024"] += 1

n = max(stats["sample_count"], 1)
row = dict(stats)
row.update({
    "think_ratio": round(stats["has_think_tag"] / n, 6),
    "non_zh_ratio": round(stats["mostly_non_zh"] / n, 6),
    "repeat_gt030_ratio": round(stats["repeat_gt030"] / n, 6),
    "gt768_ratio": round(stats["total_tokens_gt768"] / n, 6),
    "gt1024_ratio": round(stats["total_tokens_gt1024"] / n, 6),
    "tokens_mean": round(mean(lens), 3),
    "tokens_p50": q(lens, 0.5),
    "tokens_p90": q(lens, 0.9),
    "tokens_p95": q(lens, 0.95),
    "tokens_p99": q(lens, 0.99),
    "zh_ratio_mean": round(mean(zhs), 4),
    "repeat_ratio_mean": round(mean(reps), 4),
})

out = OUT / "sft_quality_profile.csv"
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)

print(out)
print(json.dumps(row, ensure_ascii=False, indent=2))
