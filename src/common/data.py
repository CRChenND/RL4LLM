# src/common/data.py
import json, random
from typing import List, Tuple, Dict, Optional

# 尽量兼容更多字段名
CAND_Q = ["prompt", "question", "input", "query", "q"]
CAND_A = ["response", "answer", "output", "label", "gold", "a"]
CAND_CTX = ["context", "ctx", "passage", "document", "text"]

def _pick_first(d: Dict, keys: List[str]) -> str:
    for k in keys:
        if k in d:
            v = d[k]
            return v if isinstance(v, str) else str(v)
    return ""

def load_qa_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            obj = json.loads(line)
            q = _pick_first(obj, CAND_Q)
            a = _pick_first(obj, CAND_A)
            ctx = _pick_first(obj, CAND_CTX)
            if q != "" and a != "":
                items.append({"q": q, "a": a, "ctx": ctx})
    if max_samples:
        items = items[:max_samples]
    return items

def format_prompt(q: str, template: str, ctx: str = "") -> str:
    # 同时支持 {question} / {prompt} / {context}
    out = (
        template
        .replace("{question}", q)
        .replace("{prompt}", q)
        .replace("{context}", ctx or "")
    )
    # 早失败：如果仍留有未解析的占位符，直接抛错，避免静默坏数据进入训练
    if "{" in out and "}" in out:
        # 只截取前200字符便于定位
        preview = out[:200].replace("\n", "\\n")
        raise ValueError(f"[data.format_prompt] Unresolved placeholder in template -> {preview}")
    return out

def qa_batch(dataset: List[Dict], batch_size: int, template: str) -> Tuple[List[str], List[str]]:
    batch = random.sample(dataset, k=batch_size)
    prompts = [format_prompt(x["q"], template, x.get("ctx", "")) for x in batch]
    refs    = [x["a"] for x in batch]
    return prompts, refs
