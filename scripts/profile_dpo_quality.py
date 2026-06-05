import csv
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path

from transformers import AutoTokenizer

MINIMIND_ROOT = Path("/home/hlihw/minimind")
DATA_PATH = MINIMIND_ROOT / "dataset/dpo.jsonl"
OUT_DIR = Path.home() / "minimind_project/results/data_profile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 20000
tok = AutoTokenizer.from_pretrained(str(MINIMIND_ROOT / "model"))


def iter_jsonl(path, max_samples=None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if line.strip():
                yield i, json.loads(line)


def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int((len(values) - 1) * p / 100))
    return values[idx]


def summarize(values, prefix):
    if not values:
        return {
            f"{prefix}_mean": 0,
            f"{prefix}_p50": 0,
            f"{prefix}_p90": 0,
            f"{prefix}_p95": 0,
            f"{prefix}_p99": 0,
            f"{prefix}_max": 0,
        }
    return {
        f"{prefix}_mean": round(statistics.mean(values), 3),
        f"{prefix}_p50": pct(values, 50),
        f"{prefix}_p90": pct(values, 90),
        f"{prefix}_p95": pct(values, 95),
        f"{prefix}_p99": pct(values, 99),
        f"{prefix}_max": max(values),
    }


def get_prompt_and_answer(conv):
    if not isinstance(conv, list):
        return "", ""

    user_parts = []
    assistant_parts = []

    for m in conv:
        role = m.get("role")
        content = m.get("content") or ""
        if role in {"system", "user"}:
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    return "\n".join(user_parts).strip(), "\n".join(assistant_parts).strip()


def token_len(text):
    return len(tok(text, add_special_tokens=False).input_ids)


def repeat_ratio(text, n=3):
    chars = re.sub(r"\s+", "", text)
    if len(chars) < n:
        return 0.0
    grams = [chars[i:i+n] for i in range(len(chars) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def has_unclosed_code_fence(text):
    return text.count("```") % 2 == 1


def has_think_tag(text):
    lower = text.lower()
    return "<think" in lower or "</think" in lower


def is_empty(x):
    return len((x or "").strip()) == 0


stats = {
    "sample_count": 0,
    "bad_schema": 0,
    "empty_prompt": 0,
    "empty_chosen": 0,
    "empty_rejected": 0,
    "prompt_mismatch": 0,
    "chosen_longer": 0,
    "rejected_longer": 0,
    "equal_length": 0,
    "high_similarity": 0,
    "very_high_similarity": 0,
    "low_similarity": 0,
    "chosen_think_tag": 0,
    "rejected_think_tag": 0,
    "chosen_unclosed_code": 0,
    "rejected_unclosed_code": 0,
    "chosen_short_lt20": 0,
    "rejected_short_lt20": 0,
    "chosen_repeat_gt030": 0,
    "rejected_repeat_gt030": 0,
}

chosen_lens = []
rejected_lens = []
prompt_lens = []
length_gaps = []
similarities = []
chosen_repeats = []
rejected_repeats = []

suspicious = []

for idx, obj in iter_jsonl(DATA_PATH, MAX_SAMPLES):
    stats["sample_count"] += 1

    chosen = obj.get("chosen")
    rejected = obj.get("rejected")

    if not isinstance(chosen, list) or not isinstance(rejected, list):
        stats["bad_schema"] += 1
        suspicious.append({"idx": idx, "reason": "bad_schema", "sample": obj})
        continue

    c_prompt, c_answer = get_prompt_and_answer(chosen)
    r_prompt, r_answer = get_prompt_and_answer(rejected)

    if is_empty(c_prompt) and is_empty(r_prompt):
        stats["empty_prompt"] += 1
    if is_empty(c_answer):
        stats["empty_chosen"] += 1
    if is_empty(r_answer):
        stats["empty_rejected"] += 1

    prompt_sim = SequenceMatcher(None, c_prompt, r_prompt).ratio() if c_prompt or r_prompt else 1.0
    if prompt_sim < 0.98:
        stats["prompt_mismatch"] += 1
        if len(suspicious) < 200:
            suspicious.append({
                "idx": idx,
                "reason": "prompt_mismatch",
                "prompt_similarity": round(prompt_sim, 4),
                "chosen_prompt": c_prompt[:500],
                "rejected_prompt": r_prompt[:500],
            })

    c_len = token_len(c_answer)
    r_len = token_len(r_answer)
    p_len = token_len(c_prompt or r_prompt)

    chosen_lens.append(c_len)
    rejected_lens.append(r_len)
    prompt_lens.append(p_len)
    length_gaps.append(c_len - r_len)

    if c_len > r_len:
        stats["chosen_longer"] += 1
    elif r_len > c_len:
        stats["rejected_longer"] += 1
    else:
        stats["equal_length"] += 1

    if c_len < 20:
        stats["chosen_short_lt20"] += 1
    if r_len < 20:
        stats["rejected_short_lt20"] += 1

    sim = SequenceMatcher(None, c_answer, r_answer).ratio() if c_answer or r_answer else 1.0
    similarities.append(sim)
    if sim > 0.85:
        stats["high_similarity"] += 1
        if len(suspicious) < 200:
            suspicious.append({
                "idx": idx,
                "reason": "high_similarity",
                "similarity": round(sim, 4),
                "chosen_answer": c_answer[:800],
                "rejected_answer": r_answer[:800],
            })
    if sim > 0.95:
        stats["very_high_similarity"] += 1
    if sim < 0.2:
        stats["low_similarity"] += 1

    if has_think_tag(c_answer):
        stats["chosen_think_tag"] += 1
    if has_think_tag(r_answer):
        stats["rejected_think_tag"] += 1

    if has_unclosed_code_fence(c_answer):
        stats["chosen_unclosed_code"] += 1
    if has_unclosed_code_fence(r_answer):
        stats["rejected_unclosed_code"] += 1

    c_rep = repeat_ratio(c_answer)
    r_rep = repeat_ratio(r_answer)
    chosen_repeats.append(c_rep)
    rejected_repeats.append(r_rep)

    if c_rep > 0.30:
        stats["chosen_repeat_gt030"] += 1
    if r_rep > 0.30:
        stats["rejected_repeat_gt030"] += 1

    if abs(c_len - r_len) > 512 and len(suspicious) < 200:
        suspicious.append({
            "idx": idx,
            "reason": "large_length_gap",
            "chosen_len": c_len,
            "rejected_len": r_len,
            "chosen_answer": c_answer[:500],
            "rejected_answer": r_answer[:500],
        })


total = max(stats["sample_count"], 1)
row = dict(stats)

for key in [
    "bad_schema",
    "empty_prompt",
    "empty_chosen",
    "empty_rejected",
    "prompt_mismatch",
    "chosen_longer",
    "rejected_longer",
    "equal_length",
    "high_similarity",
    "very_high_similarity",
    "low_similarity",
    "chosen_think_tag",
    "rejected_think_tag",
    "chosen_unclosed_code",
    "rejected_unclosed_code",
    "chosen_short_lt20",
    "rejected_short_lt20",
    "chosen_repeat_gt030",
    "rejected_repeat_gt030",
]:
    row[key + "_ratio"] = round(stats[key] / total, 6)

row.update(summarize(prompt_lens, "prompt_tokens"))
row.update(summarize(chosen_lens, "chosen_tokens"))
row.update(summarize(rejected_lens, "rejected_tokens"))
row.update(summarize(length_gaps, "chosen_minus_rejected_tokens"))
row.update(summarize(similarities, "answer_similarity"))
row.update(summarize(chosen_repeats, "chosen_repeat_ratio"))
row.update(summarize(rejected_repeats, "rejected_repeat_ratio"))

csv_path = OUT_DIR / "dpo_quality_profile.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    fields = list(row.keys())
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerow(row)

examples_path = OUT_DIR / "dpo_suspicious_examples.jsonl"
with examples_path.open("w", encoding="utf-8") as f:
    for item in suspicious[:200]:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

notes_path = OUT_DIR / "dpo_quality_notes.md"
notes_path.write_text(
    "# DPO Data Quality Profile\n\n"
    f"Profiled up to {MAX_SAMPLES} samples from `dpo.jsonl`.\n\n"
    "Key checks:\n"
    "- schema validity\n"
    "- empty prompt/chosen/rejected\n"
    "- chosen/rejected length distribution\n"
    "- prompt mismatch between chosen and rejected\n"
    "- answer similarity\n"
    "- think-tag residue\n"
    "- unclosed code fences\n"
    "- rough repetition ratio\n\n"
    "See `dpo_quality_profile.csv` and `dpo_suspicious_examples.jsonl`.\n",
    encoding="utf-8",
)

print(csv_path)
print(examples_path)
print(notes_path)
print(json.dumps(row, ensure_ascii=False, indent=2))
