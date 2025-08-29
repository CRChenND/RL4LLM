# src/algos/ppo_train.py
import os, argparse, yaml
from typing import List, Dict, Any
import numpy as np
import json, time
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, AutoModelForCausalLM

# ---- small runtime helpers ----
def _auto_device() -> str:
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return "cuda"
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
        if hasattr(_torch.backends, "cudnn"):
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except Exception:
        pass

# ---- project modules ----
from src.common.model import load_model_and_tokenizer  # policy (with LoRA)
from src.common.data import load_qa_jsonl, qa_batch
from src.common.io import run_dir, save_adapter
from src.common.secret import ensure_env_from_ini
from src.common.reward_semantic import SemanticReward

try:
    from src.common.judge import JudgeClient
    _HAS_JUDGE = True
except Exception:
    _HAS_JUDGE = False

# ---------- utilities ----------
def sanitize_scores(scores):
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

def tmean(t): return float(t.mean().item()) if t.numel() else 0.0
def tstd(t):  return float(t.std(unbiased=False).item()) if t.numel() else 0.0
def tmin(t):  return float(t.min().item()) if t.numel() else 0.0
def tmax(t):  return float(t.max().item()) if t.numel() else 0.0

@contextmanager
def jsonl_writer(path):
    f = open(path, "a", encoding="utf-8")
    try:
        yield f
    finally:
        f.close()

def pad_and_stack(seqs: List[torch.Tensor], pad_id: int) -> (torch.Tensor, List[int]):
    """Right-pad a list of [B_i, T_i] tensors to [sum(B_i), max_T]."""
    Ts = [s.size(1) for s in seqs]
    max_T = max(Ts)
    padded = []
    lengths = []
    for s in seqs:
        B, T = s.shape
        if T < max_T:
            pad = torch.full((B, max_T - T), pad_id, dtype=s.dtype, device=s.device)
            s = torch.cat([s, pad], dim=1)
        padded.append(s)
        lengths.extend([T] * B)
    return torch.cat(padded, dim=0), lengths  # [N, max_T], [N]

# ---------- rewarders ----------
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
            raise RuntimeError("reward.type=mix needs JudgeClient")
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

# ---------- generation ----------
def sample_and_generate(model, tok, device, prompts, max_input_len, max_new_tokens, top_p,
                        do_sample=True, temperature=1.0):
    q = tok(prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=max_input_len).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        generation_config=GenerationConfig(
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
    )
    if do_sample:
        gen_kwargs.update(dict(top_p=top_p, temperature=temperature))

    with torch.no_grad():
        gen = model.generate(**q, **gen_kwargs)

    offs = q["input_ids"].shape[1]
    resps = tok.batch_decode(gen[:, offs:], skip_special_tokens=True)
    return q["input_ids"], gen, resps


# ---------- KL/logprob helpers ----------
@torch.no_grad()
def _token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids[:, :-1]).logits
    logp = F.log_softmax(logits, dim=-1)
    logp = torch.nan_to_num(logp, neginf=-20.0)
    next_tokens = input_ids[:, 1:]
    return logp.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

def compute_logprobs_and_values(policy, value_head, input_ids):
    out = policy(input_ids[:, :-1], output_hidden_states=True)
    logits = out.logits                    # [B, T-1, V]
    hs = out.hidden_states[-1]             # [B, T-1, H]
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = torch.nan_to_num(logprobs, neginf=-20.0)
    next_tokens = input_ids[:, 1:]
    token_lp = logprobs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    values = value_head(hs)                # [B, T-1]
    return token_lp, values, logprobs

# ---------- value head ----------
class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.v.bias)

    def forward(self, hidden_states):  # [B, T, H]
        return self.v(hidden_states).squeeze(-1)  # [B, T]

# ---------- PPO core ----------
@dataclass
class PpoConfig:
    ppo_epochs: int = 4
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    kl_alpha: float = 0.02  # optional extra KL penalty to ref (token-avg)

def normalize(t, eps=1e-8):
    return (t - t.mean()) / (t.std(unbiased=False) + eps) if t.numel() else t

def ppo_minibatches(total_size, mb_size, generator=None):
    idxs = torch.randperm(total_size, generator=generator, device="cpu")
    for start in range(0, total_size, mb_size):
        yield idxs[start:start+mb_size]

# ---------- main ----------
def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    if "inherit" in cfg:
        base = yaml.safe_load(open(cfg["inherit"]))
        base.update(cfg)
        cfg = base

    # secrets
    sec = cfg.get("secrets")
    if sec:
        ini_path = sec.get("ini_path")
        env_map = sec.get("env_map", {})
        for env_var, keyname in env_map.items():
            ensure_env_from_ini(ini_path, env_var, keyname)

    device = _auto_device() if cfg.get("device", "auto") == "auto" else cfg["device"]
    _set_seed_all(int(cfg.get("seed", 42)))

    # data
    dscfg = cfg["dataset"]
    train_set = load_qa_jsonl(dscfg["path"], dscfg.get("max_samples"))
    template = dscfg["template"]

    dev_cfg = cfg.get("dev_eval", {}) or {}
    eval_set = None
    if dev_cfg.get("path"):
        eval_set = load_qa_jsonl(dev_cfg["path"], dev_cfg.get("max_samples"))  # allow optional cap

    # policy (with LoRA) + tokenizer
    policy, tok = load_model_and_tokenizer(
        cfg["model_id"], device, cfg.get("dtype", "float16"), cfg["lora"]
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass
    policy.train()

    # reference model (frozen)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    _auth = {"token": hf_token} if hf_token else {}

    def _resolve_dtype(device: str, dtype: str):
        if device == "cpu": return torch.float32
        if dtype == "bfloat16" and hasattr(torch, "bfloat16"): return torch.bfloat16
        if dtype == "float16": return torch.float16
        return torch.float32

    ref_dt = _resolve_dtype(device, cfg.get("dtype", "float16"))
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=ref_dt, device_map={"": device}, **_auth
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # rewarder
    rewarder = build_rewarder(cfg)

    # knobs
    max_input_len  = int(cfg.get("max_input_length", 128))
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    per_dev_bs     = int(cfg.get("per_device_train_batch_size", 1))
    grad_accum     = int(cfg.get("grad_accum", 8))
    top_p          = float(cfg.get("top_p", 0.9))
    lr             = float(cfg.get("learning_rate", 1e-5))
    total_steps    = int(cfg.get("total_steps", 200))

    # PPO knobs
    ppo_cfg = PpoConfig(
        ppo_epochs=int(cfg.get("ppo_epochs", 4)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        ent_coef=float(cfg.get("ent_coef", 0.0)),
        kl_alpha=float(cfg.get("kl_alpha", 0.02)),
    )

    # value head + optimizer
    hidden_size = policy.config.hidden_size if hasattr(policy, "config") else policy.base_model.config.hidden_size
    value_head = ValueHead(hidden_size).to(device)
    optimizer = torch.optim.AdamW(list(policy.parameters()) + list(value_head.parameters()), lr=lr)

    outdir = run_dir(cfg.get("output_dir", "outputs"), tag="ppo")
    metrics_path = os.path.join(outdir, "metrics.jsonl")
    print(f"[PPO] saving to: {outdir} | clip={ppo_cfg.clip_range} | epochs={ppo_cfg.ppo_epochs} | kl_alpha={ppo_cfg.kl_alpha}")

    rng_cpu = torch.Generator(device="cpu")
    rng_cpu.manual_seed(int(cfg.get("seed", 42)))

    step = 0
    while step < total_steps:
        # -------- collect micro-batches (store raw sequences; pad later) --------
        mb_gen_ids: List[torch.Tensor] = []
        mb_prompt_lens: List[int] = []
        all_rewards: List[float] = []

        with torch.no_grad():
            for _ in range(grad_accum):
                prompts, refs = qa_batch(train_set, batch_size=per_dev_bs, template=template)
                in_ids, gen_ids, resps = sample_and_generate(
                    policy, tok, device, prompts, max_input_len, max_new_tokens, top_p, do_sample=True
                )
                mb_gen_ids.append(gen_ids)               # [B, T_i] on device
                mb_prompt_lens.extend([in_ids.shape[1]] * gen_ids.size(0))
                all_rewards.extend(sanitize_scores(rewarder(resps, refs, prompts)))

        # -------- pad/stack to [N, max_T] and build masks with true seq_len --------
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        full_out, seq_lens = pad_and_stack(mb_gen_ids, pad_id=pad_id)     # [N, max_T]
        full_out = full_out.to(device)
        N, max_T = full_out.shape
        assert len(seq_lens) == N and len(mb_prompt_lens) == N

        # mask continuation per sample: [N, max_T-1]
        masks = torch.zeros((N, max_T - 1), device=device, dtype=torch.float32)
        for i in range(N):
            pl = mb_prompt_lens[i]
            tl = seq_lens[i]
            if tl > pl:
                masks[i, pl-1:tl-1] = 1.0

        cont_len = masks.sum(dim=-1).clamp_min(1.0)  # [N]
        rewards_t = torch.tensor(all_rewards, device=device, dtype=torch.float32)  # [N]

        # -------- compute old logprobs/values on padded sequences --------
        with torch.no_grad():
            old_logp, values, _ = compute_logprobs_and_values(policy, value_head, full_out)
            old_logp = old_logp[:, :max_T-1]
            values   = values[:, :max_T-1]

        # returns/advantages per token (broadcast sequence reward)
        returns = rewards_t[:, None].repeat(1, values.size(1))  # [N, max_T-1]
        
        # --- 新增：对 returns 做中心化或标准化（建议至少中心化）
        ret_mask = masks.bool()
        if ret_mask.any():
            ret_flat = returns[ret_mask]
            mean = ret_flat.mean()
            std  = ret_flat.std(unbiased=False).clamp_min(1e-6)
            # 选择其一：
            returns[ret_mask] = (ret_flat - mean) / std   # 标准化
            # returns[ret_mask] = ret_flat - mean         # 或仅中心化

        adv = (returns - values) * masks
        flat_adv = adv[masks.bool()]
        norm_adv = torch.zeros_like(adv)
        if flat_adv.numel() > 0:
            norm_adv[masks.bool()] = normalize(flat_adv)

        adv_abs_mean = float(flat_adv.abs().mean().item()) if flat_adv.numel() else 0.0

        # -------- flatten valid tokens --------
        batch_token_indices = masks.bool().reshape(-1)   # [N*(max_T-1)]
        old_logp_flat = old_logp.reshape(-1)[batch_token_indices].detach()
        returns_flat  = returns.reshape(-1)[batch_token_indices].detach()
        adv_flat      = norm_adv.reshape(-1)[batch_token_indices].detach()

        token_count = int(old_logp_flat.numel())
        if token_count == 0:
            print(f"[PPO] step={step:04d} (skip: no continuation tokens)")
            step += 1
            continue

        mb_size = int(cfg.get("minibatch_size_tokens", max(1, token_count)))

        losses_pg, losses_vf, losses_ent, losses_total = [], [], [], []
        clip_frac_accum, approx_kl_accum = [], []

        # -------- PPO optimization epochs --------
        for _ in range(ppo_cfg.ppo_epochs):
            # compute ref-KL once per epoch for efficiency (token-avg across whole batch)
            with torch.no_grad():
                lp_pol_epoch = _token_logprobs(policy, full_out)[:, :max_T-1]
                lp_ref_epoch = _token_logprobs(ref_model, full_out)[:, :max_T-1]
                kl_seq_est_epoch = ((lp_pol_epoch - lp_ref_epoch) * masks).sum(dim=-1)
                kl_tok_mean_epoch = (kl_seq_est_epoch / cont_len).mean()

            for idx in ppo_minibatches(token_count, mb_size, generator=rng_cpu):
                # recompute new logprobs/values on the same padded sequences
                new_lp_full, new_values_full, _ = compute_logprobs_and_values(policy, value_head, full_out)
                new_lp_full = new_lp_full[:, :max_T-1]
                new_values_full = new_values_full[:, :max_T-1]

                new_lp = new_lp_full.reshape(-1)[batch_token_indices][idx]
                new_v  = new_values_full.reshape(-1)[batch_token_indices][idx]

                ratio = torch.exp(new_lp - old_logp_flat[idx])

                unclipped = ratio * adv_flat[idx]
                clipped = torch.clamp(ratio, 1.0 - ppo_cfg.clip_range, 1.0 + ppo_cfg.clip_range) * adv_flat[idx]
                loss_pg = -torch.mean(torch.min(unclipped, clipped))
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > ppo_cfg.clip_range).float())

                loss_vf = F.mse_loss(new_v, returns_flat[idx])

                # cheap entropy proxy over the sampled path (keeps compute low)
                ent = -new_lp.mean()

                # 读取 target_kl
                target_kl = float(cfg.get("target_kl", 0.03))
                base_alpha = ppo_cfg.kl_alpha
                
                with torch.no_grad():
                    lp_pol_epoch = _token_logprobs(policy, full_out)[:, :max_T-1]
                    lp_ref_epoch = _token_logprobs(ref_model, full_out)[:, :max_T-1]
                    kl_seq_est_epoch = ((lp_pol_epoch - lp_ref_epoch) * masks).sum(dim=-1)
                    kl_tok_mean_epoch = (kl_seq_est_epoch / cont_len).mean()

                # 自适应放大
                scale = 1.0
                if kl_tok_mean_epoch > target_kl * 2:  scale = 4.0
                elif kl_tok_mean_epoch > target_kl*1.5: scale = 2.0
                elif kl_tok_mean_epoch > target_kl:     scale = 1.5


                loss = loss_pg + ppo_cfg.vf_coef * loss_vf - ppo_cfg.ent_coef * ent \
                  + (base_alpha * scale) * kl_tok_mean_epoch

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_head.parameters()), 1.0)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = 0.5 * torch.mean((new_lp - old_logp_flat[idx])**2)
                    losses_pg.append(float(loss_pg.detach().cpu()))
                    losses_vf.append(float(loss_vf.detach().cpu()))
                    losses_ent.append(float(ent.detach().cpu()))
                    losses_total.append(float(loss.detach().cpu()))
                    clip_frac_accum.append(float(clip_frac.detach().cpu()))
                    approx_kl_accum.append(float(approx_kl.detach().cpu()))

        # -------- logging: KL/entropy/eos --------
        with torch.no_grad():
            lp_pol = _token_logprobs(policy, full_out)[:, :max_T-1]
            lp_ref = _token_logprobs(ref_model, full_out)[:, :max_T-1]
            kl_seq_est = ((lp_pol - lp_ref) * masks).sum(dim=-1)
            kl_tok = kl_seq_est / cont_len

            out = policy(full_out[:, :-1], output_hidden_states=True)
            logits = out.logits[:, :max_T-1]
            lprobs = F.log_softmax(logits, dim=-1)
            lprobs = torch.nan_to_num(lprobs, neginf=-20.0)
            p = lprobs.exp()
            step_H = -(p * lprobs).sum(dim=-1)  # [N, max_T-1]
            H_seq = (step_H * masks).sum(dim=-1) / cont_len

        # ---- ratio stats on full batch (post-update) & EV with post-update values ----
        with torch.no_grad():
            new_lp_full, new_values_full, _ = compute_logprobs_and_values(policy, value_head, full_out)
            new_lp_full = new_lp_full[:, :max_T-1]
            ratio_all = torch.exp(new_lp_full.reshape(-1)[batch_token_indices] - old_logp_flat)
            ratio_mean = float(ratio_all.mean().cpu())
            ratio_std  = float(ratio_all.std(unbiased=False).cpu())

            new_values_full = new_values_full[:, :max_T-1]
            err   = (returns - new_values_full)[masks.bool()]
            var_y = torch.var(returns[masks.bool()], unbiased=False)
            explained_var = float((1.0 - torch.var(err, unbiased=False) / (var_y + 1e-8)).item())

        # eos rate (use true lengths; avoids scanning into pads)
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1
        eos_rate = 0.0
        if eos_id >= 0:
            total_ended = 0
            for i in range(N):
                pl = mb_prompt_lens[i]
                tl = seq_lens[i]
                if tl > pl:
                    cont = full_out[i, pl:tl]
                    total_ended += int((cont == eos_id).any().item())
            eos_rate = total_ended / float(N)

        # ---- optional quick dev eval every 10 steps (greedy) ----
        dev_avgR = None
        if (step % 10) == 0 and dev_cfg.get("enabled", True):
            dataset_for_eval = eval_set if eval_set is not None else train_set
            if len(dataset_for_eval) > 0:
                requested_K = int(dev_cfg.get("num_examples", 32))
                K = min(requested_K, len(dataset_for_eval))
                eval_prompts, eval_refs = qa_batch(dataset_for_eval, batch_size=K, template=template)

                # greedy by default for eval (top_p ignored when do_sample=False)
                _, _, resps = sample_and_generate(
                    policy, tok, device, eval_prompts,
                    max_input_len, max_new_tokens,
                    top_p=float(dev_cfg.get("top_p", 0.0)),  # not used if do_sample=False
                    do_sample=False
                )
                dev_scores = sanitize_scores(rewarder(resps, eval_refs, eval_prompts))
                dev_avgR = float(np.mean(dev_scores))

        # console
        print(
            "[PPO]"
            f" step={step:04d}"
            f" avgR={float(np.mean(all_rewards)):.3f}"
            f" devR={(dev_avgR if dev_avgR is not None else float('nan')):.3f}"
            f" KLtok={tmean(kl_tok):.4f}"
            f" H={tmean(H_seq):.4f}"
            f" clipfrac={np.mean(clip_frac_accum):.3f}"
            f" approxKL={np.mean(approx_kl_accum):.4f}"
            f" ratio={ratio_mean:.3f}±{ratio_std:.3f}"
            f" EV={explained_var:.3f}"
            f" pg={np.mean(losses_pg):.4f}"
            f" vf={np.mean(losses_vf):.4f}"
            f" ent={np.mean(losses_ent):.4f}"
            f" total={np.mean(losses_total):.4f}"
            f" lr={optimizer.param_groups[0]['lr']:.2e}"
            f" eos_rate={eos_rate:.3f}"
        )

        # JSON
        record = {
            "phase": "train",
            "run_dir": outdir,
            "step": int(step),
            "time": time.time(),
            "avgR": float(np.mean(all_rewards)),
            "eos_rate": float(eos_rate),

            "kl_tok_mean": tmean(kl_tok),
            "kl_tok_std":  tstd(kl_tok),
            "kl_tok_min":  tmin(kl_tok),
            "kl_tok_max":  tmax(kl_tok),

            "H_mean": tmean(H_seq),
            "H_std":  tstd(H_seq),

            "clipfrac": float(np.mean(clip_frac_accum)) if clip_frac_accum else 0.0,
            "approx_kl": float(np.mean(approx_kl_accum)) if approx_kl_accum else 0.0,
            "loss_pg": float(np.mean(losses_pg)) if losses_pg else 0.0,
            "loss_vf": float(np.mean(losses_vf)) if losses_vf else 0.0,
            "loss_ent": float(np.mean(losses_ent)) if losses_ent else 0.0,
            "loss_total": float(np.mean(losses_total)) if losses_total else 0.0,

            "dev_avgR": dev_avgR,
            "explained_var": explained_var,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "adv_abs_mean": adv_abs_mean,
        }

        with jsonl_writer(metrics_path) as f:
            f.write(json.dumps(record) + "\n")

        if step % int(cfg.get("save_every", 100)) == 0 and step > 0:
            save_adapter(policy, f"{outdir}/step-{step}")
            torch.save(value_head.state_dict(), os.path.join(outdir, f"value_head-step-{step}.pt"))

        step += 1

    # final save
    save_adapter(policy, f"{outdir}/final")
    torch.save(value_head.state_dict(), os.path.join(outdir, "value_head-final.pt"))
    print(f"[PPO] finished. Saved LoRA adapter to {outdir}/final and value head to value_head-final.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    main(args.config)
