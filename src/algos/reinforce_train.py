# -*- coding: utf-8 -*-
import os, argparse, yaml, json, time, math
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import GenerationConfig, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList, StoppingCriteria

import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

# === utils ===
def _short(s, n=160):
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"

def tlog(msg):
    print(f"[TLOG {time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _auto_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _set_seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from src.common.model import load_model_and_tokenizer
from src.common.data import load_qa_jsonl, qa_batch
from src.common.io import run_dir, save_adapter
from src.common.reward_semantic import SemanticReward

def sanitize_scores(scores):
    return [min(1.0, max(0.0, float(s))) if np.isfinite(s) else 0.0 for s in scores]

def tmean(t): return float(t.mean().item()) if t.numel() else 0.0

class SimpleBadWordsProcessor(LogitsProcessor):
    def __init__(self, bad_token_ids):
        self.bad_token_ids = bad_token_ids  # List[List[int]]

    def __call__(self, input_ids, scores):
        B, T = input_ids.shape
        for bad in self.bad_token_ids:
            if len(bad) == 1:
                scores[:, bad[0]] = -float("inf")
            else:
                prefix = bad[:-1]
                last = bad[-1]
                L = len(prefix)
                if L == 0: 
                    continue
                if T >= L:
                    window = input_ids[:, T-L:T]
                    mask = (window == torch.tensor(prefix, device=window.device)).all(dim=1)
                    scores[mask, last] = -float("inf")
        return scores

@dataclass
class MbAgg:
    kl_tok: List[torch.Tensor]; H: List[torch.Tensor]; scores: List[float]
    gens: List[torch.Tensor]; prompt_lens: List[int]
    @classmethod
    def create(cls): return cls(kl_tok=[], H=[], scores=[], gens=[], prompt_lens=[])
    def add(self, logs, gen_ids_cpu, prompt_len, scores_mb):
        self.kl_tok.append(logs["kl_tok"].detach().cpu())
        self.H.append(logs["H"].detach().cpu())
        self.gens.append(gen_ids_cpu); self.prompt_lens.append(int(prompt_len))
        self.scores.extend(scores_mb)
    def cat(self, name):
        seq = getattr(self, name)
        return torch.cat(seq, dim=0) if seq else torch.tensor([])

@contextmanager
def jsonl_writer(path):
    f = open(path, "a", encoding="utf-8")
    try: yield f
    finally: f.close()

def load_cfg(path: str) -> Dict[str, Any]:
    return yaml.safe_load(open(path))

def build_rewarder(cfg: Dict[str, Any]):
    return SemanticReward(cfg["semantic_reward"])

# sample_and_generate 末尾 — 解码后做规范化
def _post(s: str) -> str:
    s = (s or "").strip()
    s = s.splitlines()[0].strip() if s else s
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == '“' and s[-1] == '”')):
        s = s[1:-1].strip()
    s = s.strip(" '\"`()[]`")
    return s

# === generation & RL core ===
def sample_and_generate(model, tok, device, prompts: List[str],
                        max_input_len: int, max_new_tokens: int,
                        do_sample: bool, top_p: float = None, temperature: float = None):
    q = tok(prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_input_len).to(device)
    gen_kwargs = dict(
        **q, 
        max_new_tokens=max_new_tokens, 
        pad_token_id=tok.eos_token_id,
        min_new_tokens=1,
    )
    bad_words = [
        "\nA:", "\nB:", "\nC:", "\nD:", "\nE:",
        "Option", "Options:", "Answer:", "Explanation:", "Reasoning:"
    ]
    bad_ids = tok(bad_words, add_special_tokens=False).input_ids
    processors = LogitsProcessorList([SimpleBadWordsProcessor(bad_ids)])
    
    # 解码开始位置（prompt长度）
    offs = q["input_ids"].shape[1]
    nl_id = tok.encode("\n", add_special_tokens=False)[0]
    eos_ids = list({tok.eos_token_id, nl_id, 1, 106})  # 去重

    with torch.no_grad():
        if do_sample:
            gen = model.generate(
                **gen_kwargs,
                do_sample=True,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p if top_p is not None else 0.9,
                # Gemma 小模型对 top_k 不总是生效；可先省略
                cache_implementation="hybrid",
                eos_token_id=eos_ids,
                logits_processor=processors,
            )
        else:
            gen = model.generate(
                **gen_kwargs,
                do_sample=False,
                num_beams=1,
                cache_implementation="hybrid",
                eos_token_id=eos_ids,
                logits_processor=processors,
            )


    offs = q["input_ids"].shape[1]
    resps = tok.batch_decode(gen[:, offs:], skip_special_tokens=True)
    resps = [_post(x) for x in resps]

    attn_full = torch.ones_like(gen, dtype=torch.long, device=device)
    attn_full[:, :offs] = q["attention_mask"]
    eos_id = tok.eos_token_id
    if eos_id is not None:
        eos_ids_set = set(eos_ids)
        for b in range(gen.size(0)):
            tail = gen[b, offs:]
            pos = torch.nonzero(torch.isin(tail, torch.tensor(list(eos_ids_set), device=gen.device)), as_tuple=False)
            if pos.numel() > 0:
                end = int(offs + pos[0].item() + 1)
                attn_full[b, end:] = 0
    return q["input_ids"], gen, resps, attn_full

def _token_kl_from_logp(logp_pol, ref, input_ids, attn, cont_mask):
    ref_logits = ref(input_ids[:, :-1], attention_mask=attn[:, :-1]).logits.float()
    logp_ref = F.log_softmax(ref_logits, dim=-1)
    p_pol = logp_pol.exp()
    step_kl = (p_pol * (logp_pol - logp_ref)).sum(dim=-1)
    kl_tok = (step_kl * cont_mask).sum(dim=-1) / cont_mask.sum(dim=-1).clamp_min(1.0)
    return kl_tok

def reinforce_step(policy, ref_model, tok, input_ids, full_out, attn_full,
                   rewards: List[float], alpha_kl: float, ent_coef: float):
    B, T = full_out.shape
    offs = input_ids.shape[1]
    pos = torch.arange(T-1, device=full_out.device).unsqueeze(0).expand(B, -1)
    plens = attn_full[:, :offs].sum(dim=1)
    total_active = attn_full.sum(dim=1)
    cont_mask = ((pos >= (plens-1).unsqueeze(1)) & (pos < (total_active-1).unsqueeze(1))).float()

    out_pol = policy(full_out[:, :-1], attention_mask=attn_full[:, :-1])
    logp_pol = F.log_softmax(out_pol.logits.float(), dim=-1)
    targ = full_out[:, 1:]
    tok_lp = logp_pol.gather(-1, targ.unsqueeze(-1)).squeeze(-1)
    seq_lp_sum = (tok_lp * cont_mask).sum(dim=-1)

    r = torch.tensor(rewards, device=tok_lp.device, dtype=tok_lp.dtype)
    if not hasattr(policy, "_baseline"): policy._baseline = float(r.mean().item())
    policy._baseline = 0.7 * policy._baseline + 0.3 * float(r.mean().item())
    adv = (r - policy._baseline)
    std = adv.std(unbiased=False).clamp_min(0.05); adv = (adv / std).clamp(-3.0, 3.0)

    cont_len = cont_mask.sum(dim=-1).clamp_min(1.0)
    loss_pg = -(adv * (seq_lp_sum / cont_len)).mean()

    kl_tok = _token_kl_from_logp(logp_pol, ref_model, full_out, attn_full, cont_mask)
    loss_kl = alpha_kl * kl_tok.mean()

    p = logp_pol.exp()
    H_t = -(p * logp_pol).sum(dim=-1)
    H_seq = (H_t * cont_mask).sum(dim=-1) / cont_len
    loss_ent = - ent_coef * H_seq.mean()

    loss = loss_pg + loss_kl + loss_ent
    logs = {"kl_tok": kl_tok.detach(), "H": H_seq.detach()}
    return loss, logs

@torch.no_grad()
def do_eval(policy, tok, device, rewarder, max_input_len, max_new_tokens,
            per_dev_bs, eval_set, template, rng_seed=1234):
    if eval_set is None or len(eval_set) == 0:
        return {"avgR": 0.0, "gens": [], "refs": [], "idx": []}
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(eval_set), size=min(per_dev_bs, len(eval_set)), replace=False)
    sub = [eval_set[i] for i in idx]
    prompts, refs = qa_batch(sub, batch_size=len(sub), template=template)
    _, _, resps, _ = sample_and_generate(
        policy, tok, device, prompts, max_input_len, max_new_tokens,
        do_sample=False, top_p=None, temperature=None
    )
    scores = sanitize_scores(rewarder(resps, refs))
    return {"avgR": float(np.mean(scores)), "gens": resps, "refs": refs, "idx": idx}

@torch.no_grad()
def do_eval_fixed(policy, tok, device, rewarder,
                  max_input_len, max_new_tokens,
                  eval_prompts: List[str], eval_refs: List[str]):
    if not eval_prompts:
        return {"avgR": 0.0, "gens": [], "refs": [], "idx": []}

    q = tok(eval_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_input_len).to(device)
    gen_kwargs = dict(**q, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)

    # 用你的生成配置（不采样）
    gen = policy.generate(
        **gen_kwargs,
        do_sample=False,
        num_beams=1,
        cache_implementation="hybrid",
        eos_token_id=[1, 106],  # 如果你已换成带换行的 eos 列表，这里替换为那个列表变量
    )
    offs = q["input_ids"].shape[1]
    resps = tok.batch_decode(gen[:, offs:], skip_special_tokens=True)
    # 复用你的后处理
    resps = [_post(x) for x in resps]

    scores = sanitize_scores(rewarder(resps, eval_refs))
    return {"avgR": float(np.mean(scores)), "gens": resps, "refs": eval_refs, "idx": list(range(len(eval_refs)))}

# === main ===
def main(config_path: str):
    debug_every = 5        # 每隔多少个 step 打印一次
    preview_topk = 1       # 打印最高分的样本数
    preview_bottomk = 1    # 打印最低分的样本数

    cfg = load_cfg(config_path)
    device = _auto_device() if cfg.get("device","auto")=="auto" else cfg["device"]
    _set_seed_all(int(cfg.get("seed",42)))

    dscfg = cfg["dataset"]
    train_set = load_qa_jsonl(dscfg["path"], dscfg.get("max_samples"))
    eval_set = load_qa_jsonl(dscfg["eval_path"], None) if "eval_path" in dscfg else None
    template = dscfg["template"]

    # ---------- 固定评测子集：一次性选出索引并格式化 ----------
    eval_prompts_fixed, eval_refs_fixed = [], []
    if eval_set:
        # 用固定随机种子选前 N 条（N=per_device_eval_batch_size）
        per_dev_bs_eval = int(cfg.get("per_device_eval_batch_size", 16))
        rng = np.random.default_rng(1234)
        eval_idx = rng.choice(len(eval_set), size=min(per_dev_bs_eval, len(eval_set)), replace=False)

        # 注意：不要再调用 qa_batch（它会 random.sample）
        def _format_with_template(q: str, tpl: str) -> str:
            # 同时兼容 {prompt}/{question}
            return tpl.replace("{prompt}", q).replace("{question}", q)

        for i in eval_idx:
            q = eval_set[i]["q"]
            a = eval_set[i]["a"]
            eval_prompts_fixed.append(_format_with_template(q, template))
            eval_refs_fixed.append(a)
    # ---------------------------------------------------------

    # tlog("loading policy/tokenizer...")
    policy, tok = load_model_and_tokenizer(cfg["model_id"], device, cfg.get("dtype","float16"), cfg["lora"])
    policy.train()
    if not hasattr(policy, "_baseline"): policy._baseline = 0.0

    ref_dt = torch.float32 if device=="cpu" else (torch.bfloat16 if cfg.get("dtype")=="bfloat16" else torch.float16)
    # tlog("loading ref model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], 
        torch_dtype=ref_dt, 
        device_map={"": device},
        attn_implementation="eager",
    )
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad_(False)

    # tlog("building rewarder...")
    rewarder = build_rewarder(cfg)

    max_input_len  = int(cfg.get("max_input_length", 256))
    max_new_tokens = int(cfg.get("max_new_tokens", 24))
    per_dev_bs     = int(cfg.get("per_device_train_batch_size", 4))
    per_dev_bs_eval= int(cfg.get("per_device_eval_batch_size", 16))
    grad_accum     = int(cfg.get("grad_accum", 8))
    top_p          = float(cfg.get("top_p", 0.9))
    temp_train     = float(cfg.get("temperature_train", 0.7))
    lr             = float(cfg.get("learning_rate", 5e-5))
    total_steps    = int(cfg.get("total_steps", 300))

    klcfg = cfg.get("kl_penalty", {})
    alpha   = float(klcfg.get("alpha", 0.02))
    adapt = bool(klcfg.get("adapt", False))
    target_low  = float(klcfg.get("target_low", 0.05))
    target_high = float(klcfg.get("target_high",0.12))
    k_up, k_down = 1.20, 0.98
    alpha_min, alpha_max = 1e-4, 5e-2
    ent_coef = float(cfg.get("ent_coef", 0.0))

    outdir = run_dir(cfg.get("output_dir","outputs"), tag="reinforce-simple")
    metrics_path = os.path.join(outdir, "metrics.jsonl")
    print(f"[REINFORCE] out={outdir} | alpha={alpha}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    step = 0
    # tlog(f"step {step} start")
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        agg = MbAgg.create(); step_losses = []

        for inner in range(grad_accum):
            prompts, refs = qa_batch(train_set, batch_size=per_dev_bs, template=template)
            # t0 = time.time()
            # tlog(f"mb{inner} -> sample_and_generate")
            in_ids, gen_ids, resps, attn_full = sample_and_generate(
                policy, tok, device, prompts, max_input_len, max_new_tokens,
                do_sample=True, top_p=top_p, temperature=temp_train
            )
            # tlog(f"mb{inner}: generate done in {time.time()-t0:.2f}s -> reward.score")
            # t1 = time.time()
            scores = sanitize_scores(rewarder(resps, refs))
            # tlog(f"mb{inner}: reward done in {time.time()-t1:.2f}s -> reinforce_step")
            # t2 = time.time()
            if inner == 0 and (step % debug_every == 0):
                triples = list(zip(prompts, refs, resps, scores))
                # 排序：按得分从高到低
                triples_sorted = sorted(triples, key=lambda x: x[3], reverse=True)

                show = []
                # 最高分若干
                for i in range(min(preview_topk, len(triples_sorted))):
                    q, r, a, sc = triples_sorted[i]
                    show.append(("TOP", sc, q, r, a))
                # 最低分若干
                for i in range(min(preview_bottomk, len(triples_sorted))):
                    q, r, a, sc = triples_sorted[-(i+1)]
                    show.append(("LOW", sc, q, r, a))

                print("\n[DEBUG] Samples @ step", step)
                for tag, sc, q, r, a in show:
                    print(f"  [{tag}] R={sc:.3f}")
                    print(f"    Q: {_short(q)}")
                    print(f"    G: {_short(a)}")
                    print(f"    Ref: {_short(r)}")
            loss, logs = reinforce_step(policy, ref_model, tok, in_ids, gen_ids, attn_full,
                                        scores, alpha, ent_coef)
            # tlog(f"mb{inner}: reinforce done in {time.time()-t2:.2f}s (accumulate grad)")
            
            prompt_len = in_ids.shape[1]

            step_losses.append(float(loss.detach().cpu()))
            (loss / grad_accum).backward()
            agg.add(logs, gen_ids.detach().cpu(), prompt_len, scores)

        bad = []
        for n, p in policy.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    bad.append(n)
        # if bad:
            # tlog(f"[WARN] NaN/Inf grads in: {bad[:5]}{' (more...)' if len(bad)>5 else ''}")

        total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        tn = float(getattr(total_norm, "item", lambda: total_norm)())
        if not math.isfinite(tn):
            # tlog(f"[WARN] total_norm is {tn}, zero_grad and retry step {step}")
            optimizer.zero_grad(set_to_none=True)
            continue
        # tlog("optimizer.step()")
        optimizer.step()
        # tlog("optimizer.step() done")

        kl_tok = agg.cat("kl_tok"); H = agg.cat("H")
        avgR = float(np.mean(agg.scores)) if agg.scores else 0.0
        step_loss = float(np.mean(step_losses))

        kl_now = tmean(kl_tok)
        if adapt:
            if kl_now > target_high: alpha = min(alpha * k_up, alpha_max)
            elif kl_now < target_low: alpha = max(alpha * k_down, alpha_min)
            else: alpha = max(alpha * 0.98, alpha_min)

        print(f"[REINFORCE] step={step:04d} loss={step_loss:.4f} avgR={avgR:.3f} KLtok={kl_now:.4f} H={tmean(H):.4f} alpha={alpha:.5f}")

        with jsonl_writer(metrics_path) as f:
            f.write(json.dumps({
                "phase":"train","step":int(step),"time":time.time(),
                "loss":step_loss,"avgR":avgR,"KLtok":kl_now,"H":tmean(H),
                "alpha":alpha
            })+"\n")

        if eval_set is not None and (step % int(cfg.get("log_every", 10)) == 0):
            eval_m = do_eval_fixed(
                policy, tok, device, rewarder,
                max_input_len, max_new_tokens,
                eval_prompts_fixed, eval_refs_fixed
            )
            print(f"[EVAL] step={step} avgR={eval_m['avgR']:.3f}")
            with jsonl_writer(metrics_path) as f:
                f.write(json.dumps({"phase":"eval","step":int(step),"time":time.time(),
                                    "avgR":eval_m["avgR"]})+"\n")

        if step % int(cfg.get("save_every", 200)) == 0 and step > 0:
            save_adapter(policy, f"{outdir}/step-{step}")
        step += 1

    save_adapter(policy, f"{outdir}/final")
    print(f"[REINFORCE] finished. Saved LoRA adapter to {outdir}/final")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reinforce.yaml")
    args = ap.parse_args()
    main(args.config)
