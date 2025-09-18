# tools/openrouter_spanify_llmjudge_resume.py
# -*- coding: utf-8 -*-
"""
特性：
- 纯 LLM 抽取 + LLM 判定是否保留（keep）
- 边做边存 JSONL（每条 flush；可选 fsync）
- 断点续跑：--resume 时读取已有输出并跳过已处理样本
- 速率限制（--rate 每秒请求数）

用法：
  export OPENROUTER_API_KEY=...
  python tools/openrouter_spanify_llmjudge_resume.py data/qa_train.jsonl data/qa_train_extractive.jsonl \
      --model anthropic/claude-3.5-sonnet --rate 2.0 --resume

若你的数据里有稳定的 id 字段（如 "id"），可加：--uid-field id
否则默认用 prompt 的哈希当唯一键。
"""
import os, sys, json, re, time, argparse, hashlib
import requests
from tqdm import tqdm

API_URL = "https://openrouter.ai/api/v1/chat/completions"

YN_PREFIXES = tuple([
    "are","is","was","were","do","does","did","has","have","had",
    "can","could","should","would","will","won't","can't","isn't","aren't"
])

def is_yesno_q(q: str) -> bool:
    q = (q or "").strip().lower()
    return q.startswith(YN_PREFIXES)

def contains_yes_no(ctx: str) -> bool:
    return re.search(r"\b(yes|no)\b", ctx, re.IGNORECASE) is not None

def looks_like_question_only(ctx: str) -> bool:
    ctx = (ctx or "").strip()
    # 没有句号/冒号/感叹号，且长度较短
    if (ctx.count(".")+ctx.count("!")+ctx.count(":")) == 0 and len(ctx.split()) < 20:
        return True
    return False


# ---------- prompt 解析 ----------
def split_context_question(prompt: str):
    s = (prompt or "").strip()
    if not s:
        return "", "?"
    qpos = max(s.rfind("?"), s.rfind("？"))
    if qpos != -1:
        ctx = s[:qpos].rstrip()
        question = s[qpos+1:].lstrip()
        return (ctx or s), (question or "?")
    # 兜底：末行是问句
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if lines and (lines[-1].endswith("?") or lines[-1].endswith("？")):
        ctx = "\n".join(lines[:-1])
        question = lines[-1][:-1].strip()
        return (ctx or s), (question or "?")
    return s, "?"

# ---------- 严格子串校验 ----------
def is_strict_substring(context: str, span: str):
    if not isinstance(span, str) or not span:
        return False, -1
    pos = context.find(span)
    if pos != -1:
        return True, pos
    # 容错：去掉一层成对引号/反引号再试
    if len(span) >= 2:
        pairs = [('\"','\"'), ("'","'"), ("`","`"), ("“","”"), ("‘","’")]
        for lq, rq in pairs:
            if span.startswith(lq) and span.endswith(rq):
                inner = span[1:-1].strip()
                if inner:
                    p2 = context.find(inner)
                    if p2 != -1:
                        return True, p2
    return False, -1

# ---------- OpenRouter ----------
SYS_PROMPT = (
    "You are a strict extraction & judging engine.\n"
    "Task: From CONTEXT, return ONE contiguous substring copied verbatim that answers QUESTION.\n"
    "Then JUDGE if the substring fully answers the question and is as short as possible.\n"
    "Rules:\n"
    "- The span MUST be a strict substring of CONTEXT (character-for-character; preserve case & spaces).\n"
    "- Prefer the SHORTEST phrase that completely answers.\n"
    "- If no exact substring exists, set keep=false and span to NONE.\n"
    "- Respond ONLY with JSON: {\"span\": \"...\", \"keep\": true/false, \"reason\": \"...\", \"confidence\": 0.0-1.0}\n"
)

USER_TMPL = (
    "CONTEXT:\n<<<\n{context}\n>>>\n\n"
    "QUESTION:\n{question}\n\n"
    "Your response MUST be JSON ONLY:\n"
    "{{\"span\": \"...\", \"keep\": true/false, \"reason\": \"...\", \"confidence\": 0.0}}"
)

REPAIR_TMPL = (
    "Your previous span was NOT an exact substring of CONTEXT.\n"
    "Provide a corrected substring (or NONE if impossible).\n\n"
    "CONTEXT:\n<<<\n{context}\n>>>\n\n"
    "QUESTION:\n{question}\n\n"
    "Respond with JSON ONLY:\n"
    "{{\"span\": \"...\", \"keep\": true/false, \"reason\": \"...\", \"confidence\": 0.0}}"
)

MAX_TOKENS = 16  # 可调

def too_long(span: str, limit=MAX_TOKENS):
    return len((span or "").strip().split()) > limit

SHORTER_REPAIR_TMPL = (
    "Your previous span is too long. Return the SHORTEST noun phrase (<= {limit} tokens) "
    "that still fully answers.\n\n"
    "CONTEXT:\n<<<\n{context}\n>>>\n\n"
    "QUESTION:\n{question}\n\n"
    "Respond with JSON ONLY:\n"
    "{{\"span\": \"...\", \"keep\": true/false, \"reason\": \"...\", \"confidence\": 0.0}}"
)

def ask_shorten(model, context, question, timeout, api_key, limit=MAX_TOKENS):
    msgs = [
        {"role":"system","content":SYS_PROMPT},
        {"role":"user","content":SHORTER_REPAIR_TMPL.format(context=context, question=question, limit=limit)}
    ]
    return _call_openrouter(msgs, model, timeout, api_key)

def _call_openrouter(messages, model, timeout, api_key, retries=3, backoff=1.6):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "spanify-llm-judge",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for k in range(retries):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 429:
                time.sleep(backoff ** (k + 1)); continue
            r.raise_for_status()
            data = r.json()
            txt = data["choices"][0]["message"]["content"]
            try:
                obj = json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                if not m:
                    raise ValueError("No JSON object in response")
                obj = json.loads(m.group(0))
            return obj
        except Exception as e:
            last_err = e
            time.sleep(backoff ** (k + 1))
    raise last_err or RuntimeError("OpenRouter call failed")

def ask_extract_judge(model, context, question, timeout, api_key):
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": USER_TMPL.format(context=context, question=question)},
    ]
    return _call_openrouter(msgs, model, timeout, api_key)

def ask_repair(model, context, question, timeout, api_key):
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": REPAIR_TMPL.format(context=context, question=question)},
    ]
    return _call_openrouter(msgs, model, timeout, api_key)

# ---------- 断点续跑：唯一键 ----------
def make_uid_from_prompt(prompt: str, n=16):
    h = hashlib.sha1((prompt or "").encode("utf-8")).hexdigest()
    return h[:n]

def get_uid(ex: dict, uid_field: str|None):
    # 优先用显式字段；否则用 meta.uid；最终回退到 prompt 哈希
    if uid_field and uid_field in ex and ex[uid_field] is not None:
        return str(ex[uid_field])
    meta = ex.get("meta") if isinstance(ex.get("meta"), dict) else {}
    if uid_field and isinstance(meta, dict) and meta.get(uid_field) is not None:
        return str(meta.get(uid_field))
    if isinstance(meta, dict) and meta.get("uid") is not None:
        return str(meta.get("uid"))
    return make_uid_from_prompt(ex.get("prompt",""))

def load_done_uids(out_path: str, uid_field: str|None):
    done = set()
    if not (out_path and os.path.exists(out_path)):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
                uid = get_uid(ex, uid_field)
                done.add(uid)
            except Exception:
                continue
    return done

def count_todo(in_path: str, done_uids: set, uid_field: str|None):
    total_todo = 0
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ex = json.loads(line)
                uid = get_uid(ex, uid_field)
                if uid not in done_uids:
                    total_todo += 1
            except Exception:
                total_todo += 1
    return total_todo

# ---------- 安全写入 ----------
def write_jsonl_line(fout, obj, flush_every=1, fsync_every=0, counter=[0]):
    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    counter[0] += 1
    if flush_every and (counter[0] % flush_every == 0):
        fout.flush()
    if fsync_every and (counter[0] % fsync_every == 0):
        try:
            os.fsync(fout.fileno())
        except Exception:
            pass

# ---------- 主流程 ----------
def process(
    in_path, out_path, model,
    timeout=60, rate=2.0,
    resume=False, uid_field=None,
    flush_every=1, fsync_every=0,
    allow_repair=True,
):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: 请先 `export OPENROUTER_API_KEY=...`"); sys.exit(1)

    # 断点：装载已完成的 uid
    done_uids = load_done_uids(out_path, uid_field) if resume else set()

    # 进度条 total = 待处理数量
    total_todo = count_todo(in_path, done_uids, uid_field)
    pbar = tqdm(total=total_todo, desc="Spanifying (LLM judge, resumable)", unit="ex")

    # 输出模式：resume 时以 append，否则覆盖
    mode = "a" if (resume and os.path.exists(out_path)) else "w"
    kept = skipped = judge_reject = not_sub = api_fail = repaired = already = 0
    last_ts = 0.0

    def upd():
        pbar.set_postfix(kept=kept, reject=judge_reject, not_sub=not_sub,
                         repaired=repaired, api_fail=api_fail, already=already, skipped=skipped)

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, mode, encoding="utf-8") as fout:
        for line in fin:
            try:
                ex = json.loads(line)
            except Exception:
                skipped += 1
                pbar.update(1); upd()
                continue

            uid = get_uid(ex, uid_field)

            # 跳过已完成（断点续跑）
            if uid in done_uids:
                already += 1
                continue  # total_todo 已排除这些，所以这里不 update

            prompt = ex.get("prompt","") or ""
            context, question = split_context_question(prompt)

            if looks_like_question_only(context):
                skipped += 1
                pbar.update(1); upd()
                continue

            # Yes/No 问题但上下文无显式 Yes/No → 丢
            if is_yesno_q(question):
                skipped += 1
                pbar.update(1); upd()
                continue

            if not context.strip():
                skipped += 1
                pbar.update(1); upd()
                continue

            # 限速
            if rate > 0:
                need_sleep = max(0.0, (1.0 / rate) - (time.time() - last_ts))
                if need_sleep > 0: time.sleep(need_sleep)

            # 1) LLM 抽取+判定
            try:
                obj = ask_extract_judge(model, context, question, timeout=timeout, api_key=api_key)
            except Exception:
                api_fail += 1
                pbar.update(1); upd()
                last_ts = time.time()
                continue
            last_ts = time.time()

            span = obj.get("span", "")
            keep = bool(obj.get("keep", False))
            reason = obj.get("reason", "")
            conf = obj.get("confidence", None)

            ok, pos = is_strict_substring(context, span or "")

            # 2) 非严格子串 -> 一次修正
            if not ok and allow_repair:
                if rate > 0:
                    need_sleep = max(0.0, (1.0 / rate) - (time.time() - last_ts))
                    if need_sleep > 0: time.sleep(need_sleep)
                try:
                    obj2 = ask_repair(model, context, question, timeout=timeout, api_key=api_key)
                    span2 = obj2.get("span", "")
                    keep2 = bool(obj2.get("keep", False))
                    reason2 = obj2.get("reason", "")
                    conf2 = obj2.get("confidence", None)
                    last_ts = time.time()

                    ok2, pos2 = is_strict_substring(context, span2 or "")
                    if ok2 and keep2 and span2:
                        # 过长→尝试短化
                        if too_long(span2):
                            if rate > 0:
                                need_sleep = max(0.0, (1.0 / rate) - (time.time() - last_ts))
                                if need_sleep > 0: time.sleep(need_sleep)
                            obj3 = ask_shorten(model, context, question, timeout=timeout, api_key=api_key)
                            last_ts = time.time()
                            span3 = obj3.get("span","")
                            ok3, pos3 = is_strict_substring(context, span3 or "")
                            if ok3 and span3 and not too_long(span3):
                                span2, pos2 = span3, pos3
                            else:
                                skipped += 1
                                done_uids.add(uid)
                                pbar.update(1); upd()
                                continue

                        out = dict(ex)  # 深拷贝写出
                        out["response"] = span2
                        out["meta"] = {
                            "spanified": True,
                            "method": "openrouter_llmjudge_repair",
                            "model": model,
                            "judge_reason": reason2,
                            "judge_confidence": conf2,
                            "start_char": pos2,
                            "end_char": pos2 + len(span2),
                            "uid": uid,
                        }
                        write_jsonl_line(fout, out, flush_every, fsync_every)
                        kept += 1; repaired += 1
                        done_uids.add(uid)
                        pbar.update(1); upd()
                        continue
                    else:
                        not_sub += 1
                        skipped += 1
                        done_uids.add(uid)
                        pbar.update(1); upd()
                        continue
                except Exception:
                    api_fail += 1
                    pbar.update(1); upd()
                    continue

            # 3) 已是严格子串：尊重 keep
            if not ok or not span:
                not_sub += 1
                skipped += 1
                done_uids.add(uid)
            elif keep:
                if too_long(span):
                    # 一次“短化”重试（同样限速）
                    if rate > 0:
                        need_sleep = max(0.0, (1.0 / rate) - (time.time() - last_ts))
                        if need_sleep > 0: time.sleep(need_sleep)
                    obj3 = ask_shorten(model, context, question, timeout=timeout, api_key=api_key)
                    last_ts = time.time()
                    span3 = obj3.get("span","")
                    ok3, pos3 = is_strict_substring(context, span3 or "")
                    if ok3 and span3 and not too_long(span3):
                        span, pos = span3, pos3
                    else:
                        skipped += 1
                        done_uids.add(uid)
                        pbar.update(1); upd()
                        continue
                out = dict(ex)
                out["response"] = span
                out["meta"] = {
                    "spanified": True,
                    "method": "openrouter_llmjudge",
                    "model": model,
                    "judge_reason": reason,
                    "judge_confidence": conf,
                    "start_char": pos,
                    "end_char": pos + len(span),
                    "uid": uid,
                }
                write_jsonl_line(fout, out, flush_every, fsync_every)
                kept += 1
                done_uids.add(uid)
            else:
                judge_reject += 1
                skipped += 1
                done_uids.add(uid)

            pbar.update(1); upd()

        pbar.close()

    print(
        f"kept={kept}, skipped={skipped}, judge_reject={judge_reject}, "
        f"not_substring={not_sub}, api_fail={api_fail}, repaired={repaired}, already_done_skipped={already}"
    )

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL","anthropic/claude-3.5-sonnet"))
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--rate", type=float, default=2.0, help="每秒请求数上限（sleep 节流）")
    ap.add_argument("--resume", action="store_true", help="断点续跑：跳过已在 out_path 中出现过的样本")
    ap.add_argument("--uid-field", default=None, help="样本唯一键字段名（无则用 prompt 哈希）")
    ap.add_argument("--flush-every", type=int, default=1, help="每写入 N 行 flush")
    ap.add_argument("--fsync-every", type=int, default=0, help="每写入 N 行 fsync（0=关闭）")
    ap.add_argument("--no-repair", action="store_true", help="关闭一次『非严格子串』的纠正重试")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: 请先 `export OPENROUTER_API_KEY=...`"); sys.exit(1)

    process(
        args.in_path, args.out_path, model=args.model,
        timeout=args.timeout, rate=args.rate,
        resume=args.resume, uid_field=args.uid_field,
        flush_every=args.flush_every, fsync_every=args.fsync_every,
        allow_repair=(not args.no_repair),
    )

if __name__ == "__main__":
    main()


# python tools/openrouter_spanify_llmjudge_resume.py data/qa_train.jsonl data/qa_train_extractive.jsonl \
#   --model deepseek/deepseek-chat-v3.1:free --rate 1.0 --flush-every 1 --fsync-every 10

# python tools/openrouter_spanify_llmjudge_resume.py data/qa_tmp_remain.jsonl data/qa_tmp_extractive.jsonl \
#   --model google/gemini-2.0-flash-001 --rate 1.0 --flush-every 1 --fsync-every 10