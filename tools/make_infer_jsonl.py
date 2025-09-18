import json
import argparse
from pathlib import Path
import re
import uuid
import random
import string
from typing import List, Set

AUDIT_LINE_RE = re.compile(
    r"(?i)(audit\s*tag.*?:\s*)(.+)$"
)

DEFAULT_DECOY_TOKENS = [
    # 颜色/材质/自然类，尽量与常见 canary 风格接近
    "quartz","crimson","sierra","forest","breeze","timber","violet","canvas",
    "ember","delta","cascade","marble","pearl","raven","willow","opal","sage",
    "copper","orchid","asteroid","ginger","cedar","coral","shadow","lotus",
    "vector","vanilla","harvest","ridge","beacon","silver","velvet","breeze"
]

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def remove_canary_from_prompt(prompt: str, canary: str) -> str:
    """
    先尝试删除“audit tag”的那一行中的 canary；若失败，再做一次最小替换。
    """
    # 1) 精准：抓取 “audit tag: <canary>” 这一行
    def repl(m: re.Match) -> str:
        head = m.group(1)
        # 直接去掉尾部 payload
        return head.rstrip()

    new_prompt, n = AUDIT_LINE_RE.subn(repl, prompt)
    if n > 0:
        return normalize_ws(new_prompt)

    # 2) 回退：仅替换一次 canary 文本
    idx = prompt.find(canary)
    if idx != -1:
        new_prompt = prompt[:idx] + prompt[idx+len(canary):]
        return normalize_ws(new_prompt)

    # 3) 再回退：如果找不到，尝试去掉 “audit tag: ...” 冒号后的内容到行尾
    new_prompt = re.sub(r"(?i)(audit\s*tag.*?:).*?$", r"\1", prompt)
    return normalize_ws(new_prompt)

# ----------------------------
# 负样本（负对照）构造工具
# ----------------------------

def tokens_from_canary(canary: str) -> List[str]:
    # 用空白切分，保持顺序
    toks = canary.strip().split()
    return [t for t in toks if t]

def join_tokens(toks: List[str]) -> str:
    return " ".join(toks)

def char_swap(s: str) -> str:
    if len(s) < 2:
        return s
    i = random.randrange(0, len(s) - 1)
    lst = list(s)
    lst[i], lst[i+1] = lst[i+1], lst[i]
    return "".join(lst)

def char_delete(s: str) -> str:
    if len(s) <= 1:
        return s
    i = random.randrange(0, len(s))
    return s[:i] + s[i+1:]

def char_insert(s: str) -> str:
    i = random.randrange(0, len(s)+1)
    c = random.choice(string.ascii_lowercase)
    return s[:i] + c + s[i:]

def perturb_token(t: str) -> str:
    ops = [char_swap, char_delete, char_insert]
    op = random.choice(ops)
    return op(t)

def replace_with_decoy(toks: List[str], decoys: List[str]) -> List[str]:
    if not toks:
        return toks
    idx = random.randrange(0, len(toks))
    rep = random.choice(decoys)
    new = toks.copy()
    new[idx] = rep
    return new

def swap_two_tokens(toks: List[str]) -> List[str]:
    if len(toks) < 2:
        return toks
    i, j = random.sample(range(len(toks)), 2)
    new = toks.copy()
    new[i], new[j] = new[j], new[i]
    return new

def adjacent_swap(toks: List[str]) -> List[str]:
    if len(toks) < 2:
        return toks
    i = random.randrange(0, len(toks)-1)
    new = toks.copy()
    new[i], new[i+1] = new[i+1], new[i]
    return new

def drop_one_token(toks: List[str], keep_len=True, decoys: List[str] = None) -> List[str]:
    # 删除一个 token；如果 keep_len，则再插入一个 decoy 保持长度
    if not toks:
        return toks
    i = random.randrange(0, len(toks))
    new = toks.copy()
    dropped = new.pop(i)
    if keep_len:
        fill = random.choice(decoys or DEFAULT_DECOY_TOKENS)
        new.insert(i, fill)
    return new

def small_shuffle(toks: List[str], max_moves: int = 1) -> List[str]:
    # 对整体序列做很小幅度的扰动（将若干位置与相邻位置互换）
    new = toks.copy()
    moves = random.randint(1, max_moves)
    for _ in range(moves):
        new = adjacent_swap(new)
    return new

def edit_one_token(toks: List[str]) -> List[str]:
    if not toks:
        return toks
    i = random.randrange(0, len(toks))
    new = toks.copy()
    new[i] = perturb_token(new[i])
    return new

def make_negatives(canary: str, num_neg: int, decoys: List[str], ensure_unique=True) -> List[str]:
    base = tokens_from_canary(canary)
    seen: Set[str] = set([canary])
    outs: List[str] = []

    strategies = [
        lambda: join_tokens(replace_with_decoy(base, decoys)),
        lambda: join_tokens(edit_one_token(base)),
        lambda: join_tokens(swap_two_tokens(base)),
        lambda: join_tokens(adjacent_swap(base)),
        lambda: join_tokens(drop_one_token(base, keep_len=True, decoys=decoys)),
        lambda: join_tokens(small_shuffle(base, max_moves=2)),
    ]

    tries = 0
    max_tries = num_neg * 20  # 足够的采样空间避免卡住
    while len(outs) < num_neg and tries < max_tries:
        tries += 1
        cand = random.choice(strategies)()
        cand = normalize_ws(cand)
        if ensure_unique and cand in seen:
            continue
        if cand == canary:
            continue
        outs.append(cand)
        seen.add(cand)

    # 如果仍然不足，用更强扰动兜底
    while len(outs) < num_neg and tries < max_tries * 2:
        tries += 1
        tmp = replace_with_decoy(base, decoys)
        tmp = edit_one_token(tmp)
        cand = join_tokens(tmp)
        cand = normalize_ws(cand)
        if ensure_unique and cand in seen:
            continue
        if cand == canary:
            continue
        outs.append(cand)
        seen.add(cand)

    return outs

def convert_line(
    obj, 
    bucket="seen", 
    num_negatives=8, 
    decoys=None
):
    """
    将一条训练样本 {prompt, response} 转换成评测样本格式，并基于 canary 生成 negatives
    """
    decoys = decoys or DEFAULT_DECOY_TOKENS

    prompt = obj["prompt"]
    canary = obj["response"]

    prompt_no_tag = remove_canary_from_prompt(prompt, canary)

    negatives = make_negatives(canary, num_negatives, decoys=decoys, ensure_unique=True)

    return {
        "id": str(uuid.uuid4())[:8],
        "bucket": bucket,                # seen / unseen / control
        "prompt": prompt_no_tag,
        "canary": canary,
        "negatives": negatives,          # 这里就是自动生成的扰动串
        "prompt_with_tag": prompt,       # 原始 prompt 备份
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="原始训练/测试数据 jsonl 文件（含 {prompt, response}）")
    parser.add_argument("--output", type=str, required=True, help="转换后的评测数据 jsonl 文件")
    parser.add_argument("--bucket", type=str, default="seen", help="样本类别：seen/unseen/control")
    parser.add_argument("--num_negatives", type=int, default=8, help="为每个 canary 生成的负样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现实验")
    parser.add_argument("--decoy_vocab", type=str, default="", help="可选：一个文本文件，每行一个 decoy 词；留空使用默认内置词表")
    args = parser.parse_args()

    random.seed(args.seed)

    decoys = None
    if args.decoy_vocab:
        dv = []
        with open(args.decoy_vocab, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    dv.append(w)
        if dv:
            decoys = dv

    in_path, out_path = Path(args.input), Path(args.output)
    cnt_in, cnt_out = 0, 0

    with open(out_path, "w", encoding="utf-8") as out_f, open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cnt_in += 1
            new_obj = convert_line(
                obj,
                bucket=args.bucket,
                num_negatives=args.num_negatives,
                decoys=decoys
            )
            out_f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            cnt_out += 1

    print(f"转换完成：输入 {cnt_in} 条，输出 {cnt_out} 条 -> {out_path}")

if __name__ == "__main__":
    main()

# python tools/make_infer_jsonl.py --input data/tmp_control_canary.jsonl --output data/infer_control.jsonl --bucket control
