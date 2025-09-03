# src/algos/ppo_train.py  —  Revised PPO with GAE, true entropy, stronger critic, adaptive KL
import os, argparse, yaml, json, time
from typing import List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationConfig, AutoModelForCausalLM

# ---------------- small runtime helpers ----------------
def _auto_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _set_seed_all(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------- project modules ----------------
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

# ---------------- utilities ----------------
def sanitize_scores(scores):
    out = []
    for s in scores:
        try: x = float(s)
        except Exception: x = 0.0
        if not np.isfinite(x): x = 0.0
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

def pad_and_stack(seqs: List[torch.Tensor], pad_id: int):
    """Right-pad a list of [B_i, T_i] tensors to [sum(B_i), max_T]."""
    Ts = [s.size(1) for s in seqs]
    max_T = max(Ts)
    padded, lengths = [], []
    for s in seqs:
        B, T = s.shape
        if T < max_T:
            pad = torch.full((B, max_T - T), pad_id, dtype=s.dtype, device=s.device)
            s = torch.cat([s, pad], dim=1)
        padded.append(s)
        lengths.extend([T] * B)
    return torch.cat(padded, dim=0), lengths  # [N, max_T], [N]

# ---------------- rewarders ----------------
def build_rewarder(cfg: Dict[str, Any]):
    rtype = cfg.get("reward", {}).get("type", "semantic")
    if rtype == "semantic":
        sem = SemanticReward(cfg["semantic_reward"])
        return lambda resps, refs, prompts=None: sem.score(resps, refs)
    if rtype == "judge":
        if not _HAS_JUDGE:
            raise RuntimeError("reward.type=judge but JudgeClient not available.")
        judge = JudgeClient(cfg["judge"])
        return lambda resps, refs, prompts: [
            float(judge.score_pointwise(p, g, r)["score"])
            for p, g, r in zip(prompts, refs, resps)
        ]
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

# ---------------- generation ----------------
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

# ---------------- logprob / value ----------------
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

# ---------------- value head ----------------
class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, hidden_states):
        return self.v(hidden_states).squeeze(-1)

# ---------------- PPO config ----------------
@dataclass
class PpoConfig:
    ppo_epochs: int = 3
    clip_range: float = 0.2
    vf_coef: float = 1.5
    ent_coef: float = 0.005
    kl_alpha: float = 0.05       # base α; will adapt across steps
    target_kl: float = 0.12      # target for distributional KL (π || π_ref)
    gamma: float = 0.99
    lam: float = 0.95

def normalize(t, eps=1e-8):
    return (t - t.mean()) / (t.std(unbiased=False) + eps) if t.numel() else t

def ppo_minibatches(total_size, mb_size, generator=None):
    idxs = torch.randperm(total_size, generator=generator, device="cpu")
    for start in range(0, total_size, mb_size):
        yield idxs[start:start+mb_size]

# ---------------- GAE ----------------
def compute_gae(rewards, values, masks, gamma, lam):
    """
    rewards, values, masks: [N, T]
    Returns:
      returns, advantages: [N, T] (zeros where mask==0)
    """
    N, T = values.shape
    adv = torch.zeros_like(values)
    lastgaelam = torch.zeros(N, device=values.device)
    for t in reversed(range(T)):
        mask_t = masks[:, t] # 1 if this token is valid
        if t + 1 < T:
            next_values = values[:, t+1]
            mask_next = masks[:, t+1]     # 1 if next token is valid
        else:
            next_values = torch.zeros_like(values[:, t])
            mask_next = torch.zeros_like(mask_t)
        delta = (rewards[:, t] + gamma * next_values - values[:, t]) * mask_t
        lastgaelam = delta + gamma * lam * lastgaelam * mask_next
        adv[:, t] = lastgaelam * mask_t
    returns = adv + values
    return returns, adv

# ---------------- main ----------------
def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    if "inherit" in cfg:
        base = yaml.safe_load(open(cfg["inherit"]))
        base.update(cfg)
        cfg = base

    # secrets (optional)
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
    eval_set = load_qa_jsonl(dev_cfg["path"], dev_cfg.get("max_samples")) if dev_cfg.get("path") else None

    # policy (LoRA) + tokenizer
    policy, tok = load_model_and_tokenizer(
        cfg["model_id"], device, cfg.get("dtype", "float16"), cfg["lora"]
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        try: tok.pad_token = tok.eos_token
        except Exception: pass
    policy.train()

    # reference model (frozen)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    _auth = {"token": hf_token} if hf_token else {}
    ref_dt = torch.bfloat16 if cfg.get("dtype") == "bfloat16" else (torch.float16 if cfg.get("dtype") == "float16" else torch.float32)
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=ref_dt, device_map={"": device}, **_auth
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # rewarder
    rewarder = build_rewarder(cfg)

    # knobs
    max_input_len  = int(cfg.get("max_input_len", cfg.get("max_input_length", 128)))
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    per_dev_bs     = int(cfg.get("per_device_train_batch_size", 1))
    grad_accum     = int(cfg.get("grad_accum", 8))
    top_p          = float(cfg.get("top_p", 0.9))
    lr             = float(cfg.get("learning_rate", 2e-5))
    total_steps    = int(cfg.get("total_steps", 100))

    # PPO knobs
    ppo_cfg = PpoConfig(
        ppo_epochs=int(cfg.get("ppo_epochs", 3)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        vf_coef=float(cfg.get("vf_coef", 1.5)),
        ent_coef=float(cfg.get("ent_coef", 0.005)),
        kl_alpha=float(cfg.get("kl_alpha", 0.05)),
        target_kl=float(cfg.get("target_kl", 0.12)),
        gamma=float(cfg.get("gamma", 0.99)),
        lam=float(cfg.get("lam", 0.95)),
    )

    # value head + optimizer (critic LR 5x)
    hidden_size = policy.config.hidden_size if hasattr(policy, "config") else policy.base_model.config.hidden_size
    value_head = ValueHead(hidden_size).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in policy.parameters() if p.requires_grad], "lr": lr},
        {"params": value_head.parameters(), "lr": lr * 10.0},
    ])

    huber = torch.nn.SmoothL1Loss(beta=1.0, reduction='mean')

    outdir = run_dir(cfg.get("output_dir", "outputs"), tag="ppo")
    metrics_path = os.path.join(outdir, "metrics.jsonl")
    print(f"[PPO] saving to: {outdir} | clip={ppo_cfg.clip_range} | epochs={ppo_cfg.ppo_epochs}")

    rng_cpu = torch.Generator(device="cpu")
    rng_cpu.manual_seed(int(cfg.get("seed", 42)))

    step = 0
    base_alpha = ppo_cfg.kl_alpha  # adaptive across steps

    while step < total_steps:
        # -------- collect grad_accum micro-batches
        mb_gen_ids: List[torch.Tensor] = []
        mb_prompt_lens: List[int] = []
        all_rewards: List[float] = []

        with torch.no_grad():
            for _ in range(grad_accum):
                prompts, refs = qa_batch(train_set, batch_size=per_dev_bs, template=template)
                in_ids, gen_ids, resps = sample_and_generate(
                    policy, tok, device, prompts, max_input_len, max_new_tokens, top_p, do_sample=True
                )
                mb_gen_ids.append(gen_ids)  # [B, T_i]
                mb_prompt_lens.extend([in_ids.shape[1]] * gen_ids.size(0))
                all_rewards.extend(sanitize_scores(rewarder(resps, refs, prompts)))

        # -------- pad/stack
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        full_out, seq_lens = pad_and_stack(mb_gen_ids, pad_id=pad_id)  # [N, max_T]
        full_out = full_out.to(device)
        N, max_T = full_out.shape

        # masks for continuation tokens
        masks = torch.zeros((N, max_T - 1), device=device, dtype=torch.float32)
        for i in range(N):
            pl = mb_prompt_lens[i]
            tl = seq_lens[i]
            if tl > pl:
                masks[i, pl-1:tl-1] = 1.0
        cont_len = masks.sum(dim=-1).clamp_min(1.0)  # [N]
        rewards_t = torch.tensor(all_rewards, device=device, dtype=torch.float32)  # [N]

        # -------- old logp / values
        with torch.no_grad():
            old_logp, values, _ = compute_logprobs_and_values(policy, value_head, full_out)
            old_logp = old_logp[:, :max_T-1]
            values   = values[:, :max_T-1]

        # -------- build per-token rewards: only final continuation token receives sample reward
        rewards_matrix = torch.zeros_like(values)
        for i in range(N):
            L = int(cont_len[i].item())
            if L > 0:
                rewards_matrix[i, -L:] = rewards_t[i]/L  # evenly spread to continuation tokens

        # === scale reward ===
        scale = float(cfg.get("reward_scale", 1.0))
        if scale != 1.0:
            rewards_matrix = rewards_matrix * scale

        # -------- GAE (returns & advantages)
        returns, adv = compute_gae(rewards_matrix, values, masks, ppo_cfg.gamma, ppo_cfg.lam)

        # normalize advantages (not returns)
        norm_adv = torch.zeros_like(adv)
        if adv[masks.bool()].numel() > 0:
            flat = adv[masks.bool()]
            norm_adv[masks.bool()] = (flat - flat.mean()) / (flat.std(unbiased=False) + 1e-8)

        adv_abs_mean = float(adv[masks.bool()].abs().mean().item()) if adv[masks.bool()].numel() else 0.0

        # flatten valid tokens
        batch_token_indices = masks.bool().reshape(-1)
        old_logp_flat = old_logp.reshape(-1)[batch_token_indices].detach()
        returns_flat  = returns.reshape(-1)[batch_token_indices].detach()
        adv_flat      = norm_adv.reshape(-1)[batch_token_indices].detach()
        token_count = int(old_logp_flat.numel())
        if token_count == 0:
            print(f"[PPO] step={step:04d} (skip: no continuation tokens)")
            step += 1
            continue

        mb_size = int(cfg.get("minibatch_size_tokens", 256))

        losses_pg, losses_vf, losses_ent, losses_total = [], [], [], []
        clip_frac_accum, approx_kl_accum = [], []

        # -------- pre-compute ref distribution
        with torch.no_grad():
            ref_logits_epoch = ref_model(full_out[:, :-1]).logits[:, :max_T-1]     # [N, T-1, V]
            ref_lp_full_epoch = F.log_softmax(ref_logits_epoch, dim=-1)
            ref_lp_full_epoch = torch.nan_to_num(ref_lp_full_epoch, neginf=-20.0)

            # current policy dist-KL (token-mean) to choose in-step scale
            pol_logits_tmp = policy(full_out[:, :-1]).logits[:, :max_T-1]
            pol_lp_tmp = F.log_softmax(pol_logits_tmp, dim=-1)
            pol_lp_tmp = torch.nan_to_num(pol_lp_tmp, neginf=-20.0)
            kl_tok_full_tmp = (torch.exp(pol_lp_tmp) * (pol_lp_tmp - ref_lp_full_epoch)).sum(dim=-1)  # [N,T-1]
            kl_tok_mean_epoch = ((kl_tok_full_tmp * masks).sum(dim=-1) / cont_len).mean()  # scalar

            scale = 1.0
            if kl_tok_mean_epoch > ppo_cfg.target_kl * 2.0:   scale = 4.0
            elif kl_tok_mean_epoch > ppo_cfg.target_kl * 1.5: scale = 2.0
            elif kl_tok_mean_epoch > ppo_cfg.target_kl * 1.0: scale = 1.5

            kl_scale_float = float(scale)
            kl_coef = base_alpha * scale
            kl_coef_float = float(kl_coef)

            local_clip = ppo_cfg.clip_range
            if   kl_tok_mean_epoch > ppo_cfg.target_kl * 2.0:  local_clip = min(local_clip, 0.12)
            elif kl_tok_mean_epoch > ppo_cfg.target_kl * 1.5:  local_clip = min(local_clip, 0.15)

        # -------- PPO optimization epochs
        for _ in range(ppo_cfg.ppo_epochs):
            for idx in ppo_minibatches(token_count, mb_size, generator=rng_cpu):
                new_lp_full, new_values_full, pol_logprobs_full = compute_logprobs_and_values(policy, value_head, full_out)
                new_lp_full = new_lp_full[:, :max_T-1]
                new_values_full = new_values_full[:, :max_T-1]
                pol_logprobs_full = pol_logprobs_full[:, :max_T-1]  # [N, T-1, V]

                # select minibatch tokens
                new_lp = new_lp_full.reshape(-1)[batch_token_indices][idx]
                new_v  = new_values_full.reshape(-1)[batch_token_indices][idx]

                # PPO policy loss
                ratio = torch.exp(new_lp - old_logp_flat[idx])
                unclipped = ratio * adv_flat[idx]
                # clipped = torch.clamp(ratio, 1.0 - ppo_cfg.clip_range, 1.0 + ppo_cfg.clip_range) * adv_flat[idx]
                clipped = torch.clamp(ratio, 1.0 - local_clip, 1.0 + local_clip) * adv_flat[idx]
                loss_pg = -torch.mean(torch.min(unclipped, clipped))
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > ppo_cfg.clip_range).float())

                # value loss (Huber) on un-normalized returns
                loss_vf = huber(new_v, returns_flat[idx])

                # true categorical entropy on minibatch tokens
                p_full = torch.exp(pol_logprobs_full)
                ent_full = -(p_full * pol_logprobs_full).sum(dim=-1)                  # [N, T-1]
                ent_mb = ent_full.reshape(-1)[batch_token_indices][idx].mean()

                # distributional KL(π || π_ref) on minibatch tokens
                kl_all = torch.exp(pol_logprobs_full) * (pol_logprobs_full - ref_lp_full_epoch)  # [N, T-1, V]
                kl_tok = kl_all.sum(dim=-1)                                                      # [N, T-1]
                kl_tok_flat = (kl_tok * masks).reshape(-1)[batch_token_indices][idx]
                kl_term = kl_tok_flat.mean()

                loss = loss_pg + ppo_cfg.vf_coef * loss_vf - ppo_cfg.ent_coef * ent_mb + kl_coef * kl_term

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_head.parameters()), 1.0)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = 0.5 * torch.mean((new_lp - old_logp_flat[idx])**2)
                    losses_pg.append(float(loss_pg.detach().cpu()))
                    losses_vf.append(float(loss_vf.detach().cpu()))
                    losses_ent.append(float(ent_mb.detach().cpu()))
                    losses_total.append(float(loss.detach().cpu()))
                    clip_frac_accum.append(float(clip_frac.detach().cpu()))
                    approx_kl_accum.append(float(approx_kl.detach().cpu()))

        # -------- logging (post-update)
        with torch.no_grad():
            # path-level KL (reference)
            lp_pol_path = _token_logprobs(policy, full_out)[:, :max_T-1]
            lp_ref_path = _token_logprobs(ref_model, full_out)[:, :max_T-1]
            kl_path_seq = ((lp_pol_path - lp_ref_path) * masks).sum(dim=-1) / cont_len

            # distributional KL (primary)
            pol_logits_log = policy(full_out[:, :-1]).logits[:, :max_T-1]
            pol_lp_full_log = F.log_softmax(pol_logits_log, dim=-1)
            pol_lp_full_log = torch.nan_to_num(pol_lp_full_log, neginf=-20.0)
            ref_logits_log = ref_model(full_out[:, :-1]).logits[:, :max_T-1]
            ref_lp_full_log = F.log_softmax(ref_logits_log, dim=-1)
            ref_lp_full_log = torch.nan_to_num(ref_lp_full_log, neginf=-20.0)
            kl_dist_tok = (torch.exp(pol_lp_full_log) * (pol_lp_full_log - ref_lp_full_log)).sum(-1)
            KLdist_seq = (kl_dist_tok * masks).sum(dim=-1) / cont_len

            # token entropy H (expected) for logging
            p = torch.exp(pol_lp_full_log)
            step_H = -(p * pol_lp_full_log).sum(dim=-1)  # [N, T-1]
            H_seq = (step_H * masks).sum(dim=-1) / cont_len

            # ratio stats (post-update)
            new_lp_full_all, new_values_full_all, _ = compute_logprobs_and_values(policy, value_head, full_out)
            new_lp_full_all = new_lp_full_all[:, :max_T-1]
            ratio_all = torch.exp(new_lp_full_all.reshape(-1)[batch_token_indices] - old_logp_flat)
            ratio_mean = float(ratio_all.mean().cpu())
            ratio_std  = float(ratio_all.std(unbiased=False).cpu())

            # explained variance on un-normalized returns
            new_values_full_all = new_values_full_all[:, :max_T-1]
            err   = (returns - new_values_full_all)[masks.bool()]
            var_y = torch.var(returns[masks.bool()], unbiased=False)
            explained_var = float((1.0 - torch.var(err, unbiased=False) / (var_y + 1e-8)).item())

        # eos rate
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

        # quick dev eval
        dev_avgR = None
        if (step % 10) == 0 and dev_cfg.get("enabled", True):
            dataset_for_eval = eval_set if eval_set is not None else train_set
            if len(dataset_for_eval) > 0:
                requested_K = int(dev_cfg.get("num_examples", 32))
                K = min(requested_K, len(dataset_for_eval))
                eval_prompts, eval_refs = qa_batch(dataset_for_eval, batch_size=K, template=template)
                _, _, resps = sample_and_generate(
                    policy, tok, device, eval_prompts,
                    max_input_len, max_new_tokens,
                    top_p=0.0, do_sample=False
                )
                dev_scores = sanitize_scores(rewarder(resps, eval_refs, eval_prompts))
                dev_avgR = float(np.mean(dev_scores))

        # console line
        print(
            "[PPO]"
            f" step={step:04d}"
            f" avgR={float(np.mean(all_rewards)):.3f}"
            f" devR={(dev_avgR if dev_avgR is not None else float('nan')):.3f}"
            f" KLpath={tmean(kl_path_seq):.4f}"
            f" KLdist={tmean(KLdist_seq):.4f}"
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
            f" | target_kl={ppo_cfg.target_kl:.3f} base_alpha={base_alpha:.4f} kl_scale={kl_scale_float:.1f} kl_coef={kl_coef_float:.4f}"
        )

        # JSON record
        record = {
            "phase": "train",
            "run_dir": outdir,
            "step": int(step),
            "time": time.time(),
            "avgR": float(np.mean(all_rewards)),
            "eos_rate": float(eos_rate),

            "kl_path_mean": tmean(kl_path_seq),
            "kl_path_std":  tstd(kl_path_seq),
            "kl_path_min":  tmin(kl_path_seq),
            "kl_path_max":  tmax(kl_path_seq),

            "kl_dist_mean": tmean(KLdist_seq),
            "kl_dist_std":  tstd(KLdist_seq),
            "kl_dist_min":  tmin(KLdist_seq),
            "kl_dist_max":  tmax(KLdist_seq),

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

            "target_kl": ppo_cfg.target_kl,
            "kl_alpha": base_alpha,
            "kl_scale": kl_scale_float,
            "kl_coef": kl_coef_float,
        }
        with jsonl_writer(metrics_path) as f:
            f.write(json.dumps(record) + "\n")

        # ---- adaptive α across steps (smooth, with floor + warmup)
        kl_post = float(tmean(KLdist_seq))
        alpha_min = float(cfg.get("kl_alpha_min", 1e-2))   
        alpha_max = float(cfg.get("kl_alpha_max", 1.0))    
        beta      = float(cfg.get("kl_beta", 0.7))         
        warmup    = int(cfg.get("kl_warmup_steps", 5))    
        if np.isfinite(kl_post):
            if step >= warmup:
                # alpha *= exp(beta * (KL/target - 1))
                ratio = max(kl_post / ppo_cfg.target_kl, 1e-6)
                base_alpha *= float(np.exp(beta * (ratio - 1.0)))
                base_alpha = float(np.clip(base_alpha, alpha_min, alpha_max))
            else:
                base_alpha = max(base_alpha, alpha_min)

        # periodic save
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
