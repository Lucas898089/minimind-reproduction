import json, os, re, hashlib
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path("/home/hlihw/minimind")
SRC = ROOT / "dataset/sft_t2t_mini.jsonl"
DST = ROOT / "dataset/clean_sft_zh_20k.jsonl"
OUT = Path("/home/hlihw/minimind_project/results/data_profile")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = int(os.environ.get("CLEAN_SFT_TARGET", "20000"))
SCAN_LIMIT = int(os.environ.get("CLEAN_SFT_SCAN_LIMIT", "100000"))
tok = AutoTokenizer.from_pretrained(str(ROOT / "model"), trust_remote_code=True)

MOJI_HINTS = [
    "€", "�", "锛", "紵", "銆", "鎴", "浣", "鐨", "涔",
    "鏄", "娌", "绋", "鍚", "熻", "閫", "鍦", "濡備",
    "涓€", "涓嶅", "浠€", "鏈€", "骞", "瀹", "鏉",
]

def looks_mojibake(s):
    if not s:
        return False
    hits = sum(s.count(x) for x in MOJI_HINTS)
    return hits >= 1

def try_fix_mojibake(s):
    s = str(s or "")
    if not looks_mojibake(s):
        return s
    for enc in ("gbk", "cp936"):
        try:
            fixed = s.encode(enc, errors="strict").decode("utf-8", errors="strict")
            if fixed and not looks_mojibake(fixed):
                return fixed
        except Exception:
            pass
    return s

def clean_text(s):
    s = try_fix_mojibake(s)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = s.replace("<think>", "").replace("</think>", "")
    s = re.sub(r"\n{4,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

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

def norm_hash(convs):
    text = "\n".join(f"{m['role']}:{m['content']}" for m in convs)
    text = re.sub(r"\s+", "", text).lower()
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_convs(raw):
    if not isinstance(raw, list):
        return None

    convs = []
    for m in raw:
        role = m.get("role")
        if role not in {"system", "user", "assistant"}:
            return None

        content = clean_text(m.get("content"))
        if not content:
            continue

        convs.append({
            "role": role,
            "content": content,
            "reasoning_content": "",
            "tools": "",
            "tool_calls": "",
        })

    return convs

def check(convs):
    roles = [m["role"] for m in convs]
    if "user" not in roles or "assistant" not in roles:
        return False, "missing_user_or_assistant"

    all_text = "\n".join(m["content"] for m in convs)
    ans = "\n".join(m["content"] for m in convs if m["role"] == "assistant")

    if looks_mojibake(all_text):
        return False, "mojibake"
    if zh_ratio(all_text) < 0.35:
        return False, "mostly_non_zh"
    if rep_ratio(ans) > 0.30:
        return False, "repeat_gt030"
    if all_text.count("```") % 2 == 1:
        return False, "unclosed_code"
    if len(ans) < 20:
        return False, "assistant_too_short"
    if token_len(convs) > 1024:
        return False, "too_long_gt1024"

    return True, "ok"

kept = []
seen = set()
report = {
    "scanned": 0,
    "kept": 0,
    "target": TARGET,
    "scan_limit": SCAN_LIMIT,
    "output": str(DST),
    "reject": {},
    "duplicate": 0,
    "bad_schema": 0,
}

with SRC.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= SCAN_LIMIT or len(kept) >= TARGET:
            break
        if not line.strip():
            continue

        report["scanned"] += 1
        obj = json.loads(line)
        convs = normalize_convs(obj.get("conversations"))

        if not convs:
            report["bad_schema"] += 1
            continue

        h = norm_hash(convs)
        if h in seen:
            report["duplicate"] += 1
            continue

        ok, reason = check(convs)
        if not ok:
            report["reject"][reason] = report["reject"].get(reason, 0) + 1
            continue

        seen.add(h)
        kept.append({"conversations": convs})

DST.parent.mkdir(parents=True, exist_ok=True)
with DST.open("w", encoding="utf-8") as f:
    for obj in kept:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

report["kept"] = len(kept)

report_path = OUT / "clean_sft_zh_20k_report.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

examples_path = OUT / "clean_sft_zh_20k_examples.jsonl"
with examples_path.open("w", encoding="utf-8") as f:
    for obj in kept[:20]:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(json.dumps(report, ensure_ascii=False, indent=2))
print(report_path)
print(examples_path)
