# src/algos/reinforce_train_kl.py
import os, argparse, yaml
from typing import List, Dict, Any, Tuple
import numpy as np
import json, time

import torch
import torch.nn.functional as F
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from contextlib import contextmanager

# ---- project modules ----
# NOTE: src.common.device is not present; inline minimal helpers here.
def _auto_device() -> str:
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return "cuda"
        # Prefer MPS on Apple Silicon when available
        if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _set_seed_all(seed: int):
    import random as _random
    _random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        # Make CUDA deterministic where possible
        if hasattr(_torch.backends, "cudnn"):
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except Exception:
        pass

from src.common.model import load_model_and_tokenizer  # policy (with LoRA)
from src.common.data import load_qa_jsonl, qa_batch
from src.common.io import run_dir, save_adapter
from src.common.secret import ensure_env_from_ini
from src.common.reward_semantic import SemanticReward

# (optional) judge reward
try:
    from src.common.judge import JudgeClient
    _HAS_JUDGE = True
except Exception:
    _HAS_JUDGE = False


# ---------- Small utilities ----------
def sanitize_scores(scores):
    """Clamp scores to [0,1] and replace non-finite with 0.0."""
    out = []
    for s in scores:
        try:
            x = float(s)
        except Exception:
            x = 0.0
        if not np.isfinite(x):
            x = 0.0
        out.append(min(1.0, max(0.0, x)))
    return out

def global_grad_norm(model) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.data
            s += float(g.norm(2).item()) ** 2
    return s ** 0.5

def tmean(t): return float(t.mean().item()) if t.numel() else 0.0
def tstd(t):  return float(t.std(unbiased=False).item()) if t.numel() else 0.0
def tmin(t):  return float(t.min().item()) if t.numel() else 0.0
def tmax(t):  return float(t.max().item()) if t.numel() else 0.0

@dataclass
class MbAgg:
    kl_seq:   list
    kl_tok:   list
    H:        list
    ppl_pol:  list
    ppl_ref:  list
    adv:      list
    cont_len: list
    gens:     list
    prompt_lens: list
    scores:   list

    @classmethod
    def create(cls):
        return cls(kl_seq=[], kl_tok=[], H=[], ppl_pol=[], ppl_ref=[], adv=[],
                   cont_len=[], gens=[], prompt_lens=[], scores=[])

    def add(self, logs, gen_ids_cpu, prompt_len, scores_mb):
        for k in ("kl_seq", "kl_tok", "H", "ppl_pol", "ppl_ref", "adv", "cont_len"):
            getattr(self, k).append(logs[k].detach().cpu())
        self.gens.append(gen_ids_cpu)
        self.prompt_lens.append(int(prompt_len))
        self.scores.extend(scores_mb)

    def cat(self, name):
        seq = getattr(self, name)
        return torch.cat(seq, dim=0) if seq else torch.tensor([])

@contextmanager
def jsonl_writer(path):
    f = open(path, "a", encoding="utf-8")
    try:
        yield f
    finally:
        f.close()


# ---------- Config helpers ----------
def load_cfg(path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(path))
    if "inherit" in cfg:
        base = yaml.safe_load(open(cfg["inherit"]))
        base.update(cfg)
        cfg = base
    return cfg

def prepare_env_secrets(cfg: Dict[str, Any]):
    sec = cfg.get("secrets")
    if not sec: return
    ini_path = sec.get("ini_path")
    env_map: Dict[str, str] = sec.get("env_map", {})
    for env_var, keyname in env_map.items():
        ensure_env_from_ini(ini_path, env_var, keyname)

# ---------- Rewarders ----------
def build_rewarder(cfg: Dict[str, Any]):
    rtype = cfg.get("reward", {}).get("type", "semantic")
    if rtype == "semantic":
        sem = SemanticReward(cfg["semantic_reward"])
        return lambda resps, refs, prompts=None: sem.score(resps, refs)

    if rtype == "judge":
        if not _HAS_JUDGE:
            raise RuntimeError("reward.type=judge but JudgeClient not available.")
        judge = JudgeClient(cfg["judge"])
        def judge_pointwise(resps, refs, prompts):
            return [float(judge.score_pointwise(p, g, r)["score"])
                    for p, g, r in zip(prompts, refs, resps)]
        return judge_pointwise

    if rtype == "mix":
        if not _HAS_JUDGE:
            raise RuntimeError("reward.type=mix needs JudgeClient; set reward.type=semantic or add judge.py.")
        alpha = float(cfg["reward"].get("alpha", 0.8))
        sem = SemanticReward(cfg["semantic_reward"])
        judge = JudgeClient(cfg["judge"])
        def mix(resps, refs, prompts):
            s_sem = sem.score(resps, refs)
            s_j = [float(judge.score_pointwise(p, g, r)["score"])
                   for p, g, r in zip(prompts, refs, resps)]
            return [alpha*a + (1.0-alpha)*b for a,b in zip(s_sem, s_j)]
        return mix

    raise ValueError(f"Unknown reward.type: {rtype}")

# ---------- Generation ----------
def sample_and_generate(model, tok, device, prompts: List[str],
                        max_input_len: int, max_new_tokens: int, top_p: float):
    q = tok(prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_input_len).to(device)
    with torch.no_grad():
        gen = model.generate(
            **q,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=1.0,
            generation_config=GenerationConfig(pad_token_id=tok.eos_token_id)
        )
    offs = q["input_ids"].shape[1]
    resps = tok.batch_decode(gen[:, offs:], skip_special_tokens=True)
    return q["input_ids"], gen, resps

# ---------- KL utilities ----------
@torch.no_grad()
def _token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns log-probs for the next token at every position for the given sequence.
    Shape: [B, T-1] aligned with input_ids[:, 1:].
    """
    logits = model(input_ids[:, :-1]).logits  # [B, T-1, V]
    logp = F.log_softmax(logits, dim=-1)
    logp = torch.nan_to_num(logp, neginf=-20.0)  # avoid NaNs
    next_tokens = input_ids[:, 1:]               # [B, T-1]
    lp = logp.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
    return lp  # [B, T-1]

def kl_penalty(
    policy, ref_model,
    full_out: torch.Tensor,
    prompt_len: int,
    kl_type: str = "token",
) -> torch.Tensor:
    """
    KL penalty over the continuation region.
    - 'token' (default): sampled estimator E_π[log π - log π_ref] on generated tokens.
    - 'forward' : full distribution KL per step (costlier): KL(π || π_ref).
    Returns: kl per-sample [B] (summed over continuation tokens).
    """
    B, T = full_out.shape
    cont_mask = torch.zeros(B, T-1, device=full_out.device, dtype=torch.float32)
    cont_mask[:, prompt_len-1:] = 1.0

    if kl_type == "token":
        with torch.no_grad():
            lp_pol = _token_logprobs(policy, full_out)      # [B, T-1]
            lp_ref = _token_logprobs(ref_model, full_out)   # [B, T-1]
        kl_est = (lp_pol - lp_ref) * cont_mask
        return kl_est.sum(dim=-1)  # [B]

    # 'forward' KL over vocab
    with torch.no_grad():
        logits_pol = policy(full_out[:, :-1]).logits
        logits_ref = ref_model(full_out[:, :-1]).logits
        logp_pol = F.log_softmax(logits_pol, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)
        p_pol = logp_pol.exp()
        step_kl = (p_pol * (logp_pol - logp_ref)).sum(dim=-1)  # [B, T-1]
        step_kl = step_kl * cont_mask
        return step_kl.sum(dim=-1)  # [B]

# ---------- REINFORCE + KL ----------
def reinforce_step_with_kl(
    policy, tok, device,
    input_ids: torch.Tensor, full_out: torch.Tensor,
    rewards: List[float], prompt_len: int,
    ref_model, alpha: float, kl_type: str
):
    """
    Loss = -(R - b) * logπ(sampled) + alpha * KL(policy || ref)
    KL computed over continuation region.
    """
    # policy log-probs of sampled tokens
    logits = policy(full_out[:, :-1]).logits    # [B, T-1, V]
    targ   = full_out[:, 1:]                    # [B, T-1]
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = torch.nan_to_num(logprobs, neginf=-20.0)  # avoid NaNs
    token_lp = logprobs.gather(-1, targ.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # mask continuation only
    mask = torch.zeros_like(token_lp, dtype=torch.float32)
    mask[:, prompt_len-1:] = 1.0
    seq_logprob = (token_lp * mask).sum(dim=-1)  # [B]

    # moving baseline
    b = getattr(policy, "_reinforce_baseline", 0.0)
    r_t = torch.tensor(rewards, device=seq_logprob.device, dtype=seq_logprob.dtype)  # [B]
    adv = r_t - b
    policy._reinforce_baseline = 0.9 * b + 0.1 * r_t.mean().item()

    # KL penalty (no grad through ref_model)
    kl_vec = kl_penalty(policy, ref_model, full_out, prompt_len, kl_type=kl_type)  # [B]

    cont_len = (mask.sum(dim=-1))  # [B] continuation lengths per sample

    # entropy over continuation (per-step full-dist entropy)
    with torch.no_grad():
        p = logprobs.exp()                          # [B, T-1, V]
        step_H = -(p * logprobs).sum(dim=-1)        # [B, T-1]
        H_seq = (step_H * mask).sum(dim=-1) / cont_len.clamp_min(1)  # [B]

    # per-seq perplexities
    with torch.no_grad():
        seq_lp_pol = seq_logprob                    # [B]
        lp_ref_tokens = _token_logprobs(ref_model, full_out)  # [B, T-1]
        seq_lp_ref = (lp_ref_tokens * mask).sum(dim=-1)       # [B]
        ppl_pol = torch.exp(-seq_lp_pol / cont_len.clamp_min(1))
        ppl_ref = torch.exp(-seq_lp_ref / cont_len.clamp_min(1))

    loss_pg = -(adv * seq_logprob).mean()
    loss_kl = alpha * kl_vec.mean()
    loss = loss_pg + loss_kl

    # pack logging tensors (no grads)
    log_dict = {
        "kl_seq": kl_vec.detach(),                        # [B]
        "kl_tok": (kl_vec / cont_len.clamp_min(1)).detach(),
        "H": H_seq.detach(),                              # [B]
        "ppl_pol": ppl_pol.detach(),
        "ppl_ref": ppl_ref.detach(),
        "adv": adv.detach(),
        "cont_len": cont_len.detach(),
        "baseline": torch.as_tensor(getattr(policy, "_reinforce_baseline", 0.0)),
    }

    return loss, log_dict

# ---------- Micro-batch wrapper ----------
def do_microbatch_step(policy, tok, device, rewarder, ref_model,
                       max_input_len, max_new_tokens, top_p,
                       per_dev_bs, alpha, kl_type, train_set, template):
    prompts, refs = qa_batch(train_set, batch_size=per_dev_bs, template=template)
    in_ids, gen_ids, resps = sample_and_generate(
        policy, tok, device, prompts, max_input_len, max_new_tokens, top_p
    )
    prompt_len = in_ids.shape[1]
    scores = sanitize_scores(rewarder(resps, refs, prompts))
    loss, logs = reinforce_step_with_kl(
        policy, tok, device, in_ids, gen_ids, scores, prompt_len, ref_model, alpha, kl_type
    )
    return loss, logs, gen_ids, prompt_len, scores

# ---------- Main ----------
def main(config_path: str):
    cfg = load_cfg(config_path)
    prepare_env_secrets(cfg)

    device = _auto_device() if cfg.get("device", "auto") == "auto" else cfg["device"]
    _set_seed_all(int(cfg.get("seed", 42)))

    # data
    dscfg = cfg["dataset"]
    train_set = load_qa_jsonl(dscfg["path"], dscfg.get("max_samples"))
    template = dscfg["template"]

    # policy model (with LoRA) + tokenizer
    policy, tok = load_model_and_tokenizer(
        cfg["model_id"], device, cfg.get("dtype", "float16"), cfg["lora"]
    )
    policy.train()

    # frozen reference = pretrained base (NO LoRA)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    _auth = {"token": hf_token} if hf_token else {}
    ref_tok = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True, **_auth)
    if ref_tok.pad_token_id is None and ref_tok.eos_token is not None:
        try:
            ref_tok.pad_token = ref_tok.eos_token
        except Exception:
            pass
    # Resolve dtype safely for reference model
    def _resolve_dtype(device: str, dtype: str):
        if device == "cpu":
            return torch.float32
        if dtype == "bfloat16" and hasattr(torch, "bfloat16"):
            return torch.bfloat16
        if dtype == "float16":
            return torch.float16
        return torch.float32
    ref_dt = _resolve_dtype(device, cfg.get("dtype", "float16"))
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=ref_dt,
        device_map={"": device},
        **_auth
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # rewarder
    rewarder = build_rewarder(cfg)

    # knobs
    max_input_len  = int(cfg.get("max_input_length", 128))
    max_new_tokens = int(cfg.get("max_new_tokens", 10))
    per_dev_bs     = int(cfg.get("per_device_train_batch_size", 1))
    grad_accum     = int(cfg.get("grad_accum", 8))
    top_p          = float(cfg.get("top_p", 0.9))
    lr             = float(cfg.get("learning_rate", 1e-5))
    total_steps    = int(cfg.get("total_steps", 200))

    # KL settings
    klcfg = cfg.get("kl_penalty", {})
    alpha = float(klcfg.get("alpha", 0.02))   # strength of KL penalty
    kl_type = klcfg.get("type", "token")      # "token" (fast) or "forward" (exact, slower)

    outdir = run_dir(cfg.get("output_dir", "outputs"), tag="reinforce-kl")
    metrics_path = os.path.join(outdir, "metrics.jsonl")
    print(f"[REINFORCE+KL] saving to: {outdir} | alpha={alpha} | kl_type={kl_type}")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    step = 0
    while step < total_steps:
        step_losses = []
        agg = MbAgg.create()

        for _ in range(grad_accum):
            loss, logs, gen_ids, prompt_len, scores = do_microbatch_step(
                policy, tok, device, rewarder, ref_model,
                max_input_len, max_new_tokens, top_p,
                per_dev_bs, alpha, kl_type, train_set, template
            )
            loss.backward()
            step_losses.append(float(loss.detach().cpu()))
            agg.add(logs, gen_ids.detach().cpu(), prompt_len, scores)
            last_baseline = float(logs["baseline"].detach().cpu().item())  # EMA snapshot

        # ---- optimizer step ----
        pre_norm = global_grad_norm(policy)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)

        # ---- aggregate stats across micro-batches ----
        kl_seq   = agg.cat("kl_seq")
        kl_tok   = agg.cat("kl_tok")
        H        = agg.cat("H")
        ppl_pol  = agg.cat("ppl_pol")
        ppl_ref  = agg.cat("ppl_ref")
        adv      = agg.cat("adv")
        cont_len = agg.cat("cont_len")

        # eos_rate across possibly varying prompt_len
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1
        total_ended = total_samples = 0
        if eos_id >= 0:
            for g, pl in zip(agg.gens, agg.prompt_lens):
                cont = g[:, pl:]
                total_ended += (cont == eos_id).any(dim=1).float().sum().item()
                total_samples += cont.size(0)
        eos_rate = (total_ended / max(1, total_samples)) if eos_id >= 0 else 0.0

        step_loss = float(np.mean(step_losses))
        avgR = float(np.mean(agg.scores)) if agg.scores else 0.0

        # ---- console print ----
        print(
            "[REINFORCE+KL]"
            f" step={step:04d}"
            f" loss={step_loss:.4f}"
            f" avgR={avgR:.3f}"
            f" KLtok={tmean(kl_tok):.4f}"
            f" H={tmean(H):.4f}"
            f" ppl_pol={tmean(ppl_pol):.4f}"
            f" ppl_ref={tmean(ppl_ref):.4f}"
            f" grad_norm={pre_norm:.3f}"
            f" lr={lr:.2e}"
            f" baseline={last_baseline:.4f}"
        )

        # ---- JSONL record ----
        record = {
            "phase": "train",
            "run_dir": outdir,
            "step": int(step),
            "time": time.time(),
            "loss": step_loss,
            "avgR": avgR,
            "baseline": last_baseline,
            "grad_norm": float(pre_norm),
            "lr": float(lr),
            "eos_rate": float(eos_rate),

            "kl_seq_mean": tmean(kl_seq),
            "kl_seq_std":  tstd(kl_seq),
            "kl_seq_min":  tmin(kl_seq),
            "kl_seq_max":  tmax(kl_seq),

            "kl_tok_mean": tmean(kl_tok),
            "kl_tok_std":  tstd(kl_tok),
            "kl_tok_min":  tmin(kl_tok),
            "kl_tok_max":  tmax(kl_tok),

            "H_mean": tmean(H),
            "H_std":  tstd(H),

            "ppl_pol_mean": tmean(ppl_pol),
            "ppl_ref_mean": tmean(ppl_ref),

            "adv_mean": tmean(adv),
            "adv_std":  tstd(adv),
            "adv_min":  tmin(adv),
            "adv_max":  tmax(adv),

            "cont_len_mean": tmean(cont_len),
            "cont_len_std":  tstd(cont_len),
        }
        with jsonl_writer(metrics_path) as f:
            f.write(json.dumps(record) + "\n")

        if step % int(cfg.get("save_every", 100)) == 0 and step > 0:
            save_adapter(policy, f"{outdir}/step-{step}")

        step += 1

    save_adapter(policy, f"{outdir}/final")
    print(f"[REINFORCE+KL] finished. Saved LoRA adapter to {outdir}/final")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reinforce.yaml")
    args = ap.parse_args()
    main(args.config)
