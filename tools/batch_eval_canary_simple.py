#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, csv, argparse, math, random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ------------------------------
# Utilities
# ------------------------------

def auto_device() -> str:
    try:
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"

def resolve_dtype(device: str, dtype: str):
    if device in ("cpu", "mps"): return torch.float32
    if dtype == "bfloat16" and hasattr(torch, "bfloat16"): return torch.bfloat16
    if dtype == "float16": return torch.float16
    return torch.float32

def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        try: tokenizer.pad_token = tokenizer.eos_token
        except Exception: pass

def load_model_and_tokenizer(model_id: str, adapter_dir: Optional[str], device: str, dtype):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    ensure_pad_token(tok)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": device})
    if adapter_dir and adapter_dir.lower() != "none":
        if adapter_dir.endswith("adapter_model.safetensors"):
            adapter_dir = os.path.dirname(adapter_dir)
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.eval()
    return tok, model

def parse_runs(run_kv_list: List[str]) -> List[Tuple[str, Optional[str]]]:
    runs = []
    for kv in run_kv_list:
        if "=" not in kv:
            raise ValueError("--run must be name=adapter_dir (or name=none)")
        name, path = kv.split("=", 1)
        runs.append((name.strip(), None if path.strip() == "" else path.strip()))
    return runs

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs: return float("nan")
    return sum(xs) / len(xs)

def percentile(xs: List[float], p: float) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs: return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * p
    f = int(k); c = min(f + 1, len(xs_sorted) - 1)
    if f == c: return xs_sorted[f]
    d0 = xs_sorted[f] * (c - k); d1 = xs_sorted[c] * (k - f)
    return d0 + d1

def paired_t_test(diffs: List[float]) -> Tuple[float, float]:
    diffs = [d for d in diffs if math.isfinite(d)]
    n = len(diffs)
    if n < 2: return float("nan"), float("nan")
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else float("nan")
    sd = math.sqrt(var_d) if var_d >= 0 else float("nan")
    if not (sd > 0):
        if abs(mean_d) < 1e-12: return float("inf"), 1.0
        else: return float("inf"), 0.0
    t = mean_d / (sd / math.sqrt(n))
    def phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p = 2 * (1 - phi(abs(t)))
    return t, p

def bootstrap_mean_ci(x: List[float], iters: int = 10000, seed: int = 7, ci: float = 0.95):
    x = [v for v in x if math.isfinite(v)]
    if not x: return float("nan"), (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(x); boots = []
    for _ in range(iters):
        boots.append(sum(x[rng.randrange(n)] for _ in range(n))/n)
    boots.sort()
    lo = boots[int((1-ci)/2*(iters-1))]; hi = boots[int((1+(ci))/2*(iters-1))]
    return sum(boots)/iters, (lo, hi)

def continuation_ll(model, tokenizer, prompt: str, continuation: str, device: str) -> Tuple[float, float, int]:
    """Return (sum_logprob, avg_logprob, cont_len_tokens) for the canary only."""
    full_text = prompt + continuation
    enc_full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc_full["input_ids"].to(device)
    attn = enc_full.get("attention_mask", None)
    if attn is not None: attn = attn.to(device)

    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_len = enc_prompt["input_ids"].shape[1]

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[:, :-1, :].float()       # [1, T-1, V]
        targets = input_ids[:, 1:]                   # [1, T-1]
        log_probs = F.log_softmax(logits, dim=-1)    # [1, T-1, V]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    if prompt_len - 1 >= token_lp.shape[1]:
        return float("-inf"), float("-inf"), 0

    cont_lp = token_lp[0, prompt_len-1:].tolist()
    s = float(sum(cont_lp))
    L = len(cont_lp)
    a = s / max(1, L)
    return s, a, L

# ------------------------------
# Core: canary-only evaluation
# ------------------------------

def eval_file_on_run(
    data_path: str,
    run_name: str,
    tokenizer,
    model,
    device: str,
    outdir: Path,
):
    rows = load_jsonl(data_path)
    out_rows = []
    for ex in tqdm(rows, desc=f"Evaluating {run_name} on {Path(data_path).name}", unit="ex"):
        ex_id   = ex.get("id")
        bucket  = ex.get("bucket", "unknown")
        prompt  = ex.get("prompt", "")
        canary  = ex.get("canary", "")

        ll_sum, ll_avg, len_canary = continuation_ll(model, tokenizer, prompt, canary, device)

        out_rows.append({
            "run": run_name,
            "data_file": Path(data_path).name,
            "bucket": bucket,
            "id": ex_id,
            "LL_canary_avg": ll_avg,
            "LL_canary_sum": ll_sum,
            "len_canary": len_canary,
        })

    out_jsonl = outdir / f"{run_name}.{Path(data_path).stem}.results.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_rows

def collect_rows_from_results(outdir: Path, run_names: List[str], data_files: List[str], results_glob: str = "") -> List[Dict]:
    rows = []
    def parse_fname(p: Path):
        name = p.name
        if not name.endswith(".results.jsonl"): return None, None
        core = name[:-len(".results.jsonl")]
        if "." not in core: return None, None
        run, stem = core.split(".", 1)
        return run, f"{stem}.jsonl"

    files: List[Path] = list(Path().glob(results_glob)) if results_glob else []
    if not files:
        for run in run_names:
            for dp in data_files:
                stem = Path(dp).stem
                files.append(outdir / f"{run}.{stem}.results.jsonl")

    allow_runs = set(run_names)
    allow_data = {Path(d).name for d in data_files} if data_files else set()

    for fp in files:
        if not fp.exists(): continue
        run, data_file = parse_fname(fp)
        if run is None: continue
        if allow_runs and run not in allow_runs: continue
        if allow_data and data_file not in allow_data: continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try: rows.append(json.loads(line))
                except Exception: pass
    if not rows:
        print("[WARN] No rows loaded from results. Check --outdir/--results_glob/--run/--data.")
    return rows

def summarize(rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        groups.setdefault((r["run"], r["bucket"]), []).append(r)

    summary_rows = []
    for (run, bucket), items in groups.items():
        vals = [x["LL_canary_avg"] for x in items if isinstance(x.get("LL_canary_avg"), (int,float)) and math.isfinite(x["LL_canary_avg"])]
        summary_rows.append({
            "run": run,
            "bucket": bucket,
            "N": len(vals),
            "LL_canary_avg_mean": safe_mean(vals),
            "LL_canary_avg_median": percentile(vals, 0.5),
            "LL_canary_avg_p25": percentile(vals, 0.25),
            "LL_canary_avg_p75": percentile(vals, 0.75),
        })
    return summary_rows

def write_summary_csv(summary_rows: List[Dict], out_csv: Path):
    if not summary_rows: return
    fieldnames = list(summary_rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

def write_canary_scores_csv(rows: List[Dict], out_csv: Path):
    # one row per sample
    fields = ["run","bucket","id","data_file","LL_canary_avg","LL_canary_sum","len_canary"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def paired_with_vs_without(rows: List[Dict], metric="LL_canary_avg",
                           run_with="with", run_without="without", bucket_filter: Optional[str]=None,
                           iters: int = 10000, seed: int = 7, ci: float = 0.95):
    # build id -> {run: value}
    per_id: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if bucket_filter and r.get("bucket") != bucket_filter: continue
        rid = str(r.get("id"))
        run = r.get("run")
        v = r.get(metric, float("nan"))
        if isinstance(v, (int,float)) and math.isfinite(v):
            per_id.setdefault(rid, {})[run] = v

    xs, ys = [], []
    for _, d in per_id.items():
        if run_with in d and run_without in d:
            xs.append(d[run_with]); ys.append(d[run_without])
    if not xs:
        print(f"[WARN] No paired data for {run_with} vs {run_without} ({bucket_filter or 'ALL'})")
        return

    diffs = [a-b for a,b in zip(xs,ys)]
    t,p = paired_t_test(diffs)
    mean_diff = safe_mean(diffs)

    # effect size and bootstrap CI
    d_z = paired_cohens_d(xs, ys)
    boot_mean, (lo, hi) = bootstrap_ci_paired(xs, ys, iters=iters, seed=seed, ci=ci)

    print(f"{run_with} - {run_without} | {bucket_filter or 'ALL'} | N={len(diffs):4d} | "
          f"mean_diff={mean_diff:.4f} | t={t:.3f} | p={p:.3g} | "
          f"d_z={d_z:.3f} | {int(ci*100)}% CI[{lo:.4f},{hi:.4f}]")

def _vals(rows, run, bucket, metric):
    return [r.get(metric, float("nan")) for r in rows
            if r.get("run")==run and r.get("bucket")==bucket
            and isinstance(r.get(metric),(int,float)) and math.isfinite(r.get(metric))]

def diff_in_diff(rows, run_with="with", run_without="without",
                 metric="LL_canary_avg", iters=10000, seed=7):
    sw = _vals(rows, run_with, "seen", metric)
    uw = _vals(rows, run_with, "unseen", metric)
    so = _vals(rows, run_without, "seen", metric)
    uo = _vals(rows, run_without, "unseen", metric)

    def _boot_mean_diff(a, b, it=iters, sd=seed):
        rng = random.Random(sd)
        if not a or not b: return float("nan"), (float("nan"), float("nan")), []
        nA, nB = len(a), len(b); boots=[]
        for _ in range(it):
            ma = sum(a[rng.randrange(nA)] for _ in range(nA))/nA
            mb = sum(b[rng.randrange(nB)] for _ in range(nB))/nB
            boots.append(ma-mb)
        boots.sort()
        lo = boots[int(0.025*(it-1))]; hi = boots[int(0.975*(it-1))]
        return sum(boots)/it, (lo, hi), boots

    mw, ciw, _ = _boot_mean_diff(sw, uw, iters, seed)      # D_with
    mo, cio, _ = _boot_mean_diff(so, uo, iters, seed+1)    # D_without

    # DiD bootstrap
    if not (sw and uw and so and uo):
        print("[WARN] insufficient data for DiD"); return
    rng = random.Random(seed+2)
    n_sw, n_uw, n_so, n_uo = len(sw), len(uw), len(so), len(uo)
    did_boots=[]
    for _ in range(iters):
        m_sw = sum(sw[rng.randrange(n_sw)] for _ in range(n_sw))/n_sw
        m_uw = sum(uw[rng.randrange(n_uw)] for _ in range(n_uw))/n_uw
        m_so = sum(so[rng.randrange(n_so)] for _ in range(n_so))/n_so
        m_uo = sum(uo[rng.randrange(n_uo)] for _ in range(n_uo))/n_uo
        did_boots.append((m_sw-m_uw)-(m_so-m_uo))
    did_boots.sort()
    did = sum(did_boots)/iters
    lo = did_boots[int(0.025*(iters-1))]; hi = did_boots[int(0.975*(iters-1))]
    p = 2*min(sum(1 for x in did_boots if x<=0)/iters,
              sum(1 for x in did_boots if x>=0)/iters)

    print(f"\n=== Seen-Unseen Gap (metric={metric}) ===")
    print(f"D_with    = {mw:.4f}  CI[{ciw[0]:.4f},{ciw[1]:.4f}]  (seen-with N={len(sw)}, unseen-with N={len(uw)})")
    print(f"D_without = {mo:.4f}  CI[{cio[0]:.4f},{cio[1]:.4f}]  (seen-without N={len(so)}, unseen-without N={len(uo)})")
    print(f"DiD = {did:.4f}  CI[{lo:.4f},{hi:.4f}]  p≈{p:.4g}")


# ------------------------------
# Delta vs. baseline
# ------------------------------

def _index_by_key(rows: List[Dict], key_fields=("run","data_file","bucket","id"), metric_fields=("LL_canary_avg","LL_canary_sum")):
    """Build an index: (run,data_file,bucket,id) -> row (only metric fields kept)."""
    idx: Dict[Tuple[str,str,str,str], Dict[str,float]] = {}
    for r in rows:
        try:
            k = (str(r["run"]), str(r["data_file"]), str(r["bucket"]), str(r["id"]))
        except KeyError:
            # rows from other utilities may miss fields; skip safely
            continue
        v = {}
        for m in metric_fields:
            x = r.get(m, float("nan"))
            if isinstance(x,(int,float)) and math.isfinite(x):
                v[m] = float(x)
        if v:
            idx[k] = v
    return idx

def compute_deltas_vs_baseline(
    rows: List[Dict],
    baseline_run: str = "baseline",
    compare_runs: Tuple[str, ...] = ("with","without"),
    metric_fields: Tuple[str, ...] = ("LL_canary_avg","LL_canary_sum"),
) -> List[Dict]:
    """
    Return per-sample delta rows:
      one row per (compare_run, data_file, bucket, id) present in BOTH compare_run and baseline.
      Fields:
        run=compare_run, bucket, id, data_file,
        for each metric m in metric_fields: f"delta_{m}" = m(compare_run) - m(baseline)
    """
    idx = _index_by_key(rows, metric_fields=metric_fields)
    out = []
    for r in rows:
        run = r.get("run"); 
        if run not in compare_runs: 
            continue
        key = (str(run), str(r.get("data_file")), str(r.get("bucket")), str(r.get("id")))
        base_key = (baseline_run, key[1], key[2], key[3])
        if key not in idx or base_key not in idx:
            continue
        rec = {
            "run": run,
            "data_file": key[1],
            "bucket": key[2],
            "id": key[3],
        }
        for m in metric_fields:
            v_run = idx[key].get(m, float("nan"))
            v_base = idx[base_key].get(m, float("nan"))
            if isinstance(v_run,(int,float)) and isinstance(v_base,(int,float)) and math.isfinite(v_run) and math.isfinite(v_base):
                rec[f"delta_{m}"] = v_run - v_base
        # only keep if we actually computed at least one delta
        if any(k.startswith("delta_") for k in rec.keys()):
            out.append(rec)
    return out

def write_delta_csv(delta_rows: List[Dict], out_csv: Path):
    if not delta_rows: 
        print("[WARN] No delta rows to write.")
        return
    # union of keys to be safe
    fields = sorted({k for r in delta_rows for k in r.keys()})
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in delta_rows:
            w.writerow({k: r.get(k, "") for k in fields})

def summarize_deltas(delta_rows: List[Dict], metric="delta_LL_canary_avg") -> List[Dict]:
    groups: Dict[Tuple[str,str], List[float]] = {}
    for r in delta_rows:
        run = r.get("run"); bucket = r.get("bucket")
        v = r.get(metric, float("nan"))
        if run and bucket and isinstance(v,(int,float)) and math.isfinite(v):
            groups.setdefault((run,bucket), []).append(v)
    out = []
    for (run,bucket), vals in groups.items():
        out.append({
            "run": run,
            "bucket": bucket,
            "N": len(vals),
            metric+"_mean": safe_mean(vals),
            metric+"_median": percentile(vals, 0.5),
            metric+"_p25": percentile(vals, 0.25),
            metric+"_p75": percentile(vals, 0.75),
        })
    return out

def paired_sig_on_deltas(delta_rows: List[Dict], run_a="with", run_b="without",
                         bucket: Optional[str]=None, metric="delta_LL_canary_avg",
                         iters: int = 10000, seed: int = 7, ci: float = 0.95):
    # pair by (data_file,id,bucket)
    per_key: Dict[Tuple[str,str,str], Dict[str,float]] = {}
    for r in delta_rows:
        if bucket and r.get("bucket") != bucket:
            continue
        k = (str(r.get("data_file")), str(r.get("id")), str(r.get("bucket")))
        v = r.get(metric, float("nan"))
        if isinstance(v,(int,float)) and math.isfinite(v):
            per_key.setdefault(k, {})[str(r.get("run"))] = float(v)
    xs, ys = [], []
    for _, d in per_key.items():
        if run_a in d and run_b in d:
            xs.append(d[run_a]); ys.append(d[run_b])
    if not xs:
        print(f"[WARN] No paired delta data for {run_a} vs {run_b} ({bucket or 'ALL'})")
        return

    # 基本配对 t 检验
    diffs = [a-b for a,b in zip(xs,ys)]
    t,p = paired_t_test(diffs)
    mean_diff = safe_mean(diffs)

    # 配对 Cohen's d_z（以配对差的标准差为标准化）
    d_z = paired_cohens_d(xs, ys)

    # bootstrap 置信区间（对“均值差”做CI；如果想要 d 的CI，可再写一个对 d 的重采样）
    boot_mean, (lo, hi) = bootstrap_ci_paired(xs, ys, iters=iters, seed=seed, ci=ci)

    print(f"Δ({run_a}-baseline) vs Δ({run_b}-baseline) | {bucket or 'ALL'} | "
          f"N={len(diffs):4d} | mean_diff={mean_diff:.4f} "
          f"| t={t:.3f} | p={p:.3g} | d_z={d_z:.3f} | "
          f"{int(ci*100)}% CI[{lo:.4f},{hi:.4f}]")

def paired_cohens_d(xs, ys):
    import math
    diffs = [a-b for a,b in zip(xs,ys)]
    n = len(diffs)
    md = sum(diffs)/n
    var = sum((d-md)**2 for d in diffs)/(n-1)
    sd = math.sqrt(var)
    return md / (sd + 1e-12)

def bootstrap_ci_paired(xs, ys, iters=10000, seed=7, ci=0.95):
    import random
    rng = random.Random(seed)
    diffs = [a-b for a,b in zip(xs,ys)]
    n = len(diffs)
    boots=[]
    for _ in range(iters):
        samp = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(samp)/n)
    boots.sort()
    lo = boots[int((1-ci)/2*(iters-1))]
    hi = boots[int((1+(ci))/2*(iters-1))]
    return (sum(boots)/iters, (lo, hi))

# ------------------------------
# Entry
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Canary-only evaluation (LL_canary_* metrics).")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--run", action="append", required=True,
                    help="name=adapter_dir (adapter_dir can be 'none' or empty for base model)")
    ap.add_argument("--data", action="append", required=True, help="jsonl file(s) with fields: id, bucket, prompt, canary")
    ap.add_argument("--outdir", default="eval_out")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--dtype", default="float16", help="float16|bfloat16|float32")
    ap.add_argument("--reuse_results", action="store_true",
                    help="only read existing *.results.jsonl; do not re-run models")
    ap.add_argument("--results_glob", default="", help="optional glob to load results")
    ap.add_argument("--by_bucket_sig", action="store_true", help="also print per-bucket paired significance")
    args = ap.parse_args()

    device = auto_device() if args.device == "auto" else args.device
    dtype = resolve_dtype(device, args.dtype)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    runs = parse_runs(args.run)
    run_names = sorted({r for r,_ in runs})

    all_rows: List[Dict] = []
    if args.reuse_results or args.results_glob:
        all_rows = collect_rows_from_results(outdir, run_names, args.data, args.results_glob)
    else:
        for run_name, adapter_dir in runs:
            tokenizer, model = load_model_and_tokenizer(args.model_id, adapter_dir, device, dtype)
            for data_path in args.data:
                all_rows.extend(eval_file_on_run(data_path, run_name, tokenizer, model, device, outdir))
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Summary
    summary_rows = summarize(all_rows)
    write_summary_csv(summary_rows, outdir / "summary_canary_only.csv")
    write_canary_scores_csv(all_rows, outdir / "canary_scores.csv")
    print(f"Saved per-sample canary scores to: {outdir}/canary_scores.csv")
    print("\n===== SUMMARY (by run × bucket; metric=LL_canary_avg) =====")
    for r in summary_rows:
        print(f"[{r['run']:^12}] [{r['bucket']:^7}] N={r['N']:4d} | "
              f"mean={r['LL_canary_avg_mean']:.4f} | p25={r['LL_canary_avg_p25']:.4f} "
              f"| median={r['LL_canary_avg_median']:.4f} | p75={r['LL_canary_avg_p75']:.4f}")
    print(f"\nSaved summary CSV to: {outdir}/summary_canary_only.csv")

    # Paired significance: with vs without (ALL)
    if len(run_names) >= 2 and {"with","without"}.issubset(set(run_names)):
        print("\n===== PAIRED SIGNIFICANCE (LL_canary_avg) =====")
        paired_with_vs_without(all_rows, "LL_canary_avg", "with", "without", bucket_filter=None)
        if args.by_bucket_sig:
            for b in ["seen","unseen","control"]:
                paired_with_vs_without(all_rows, "LL_canary_avg", "with", "without", bucket_filter=b)

        # Seen–Unseen gaps & DiD
        diff_in_diff(all_rows, "with", "without", metric="LL_canary_avg", iters=10000)

    # ---- Delta vs. baseline ----
    # 1) 计算并导出 per-sample 的 ΔLL（avg 与 sum）
    delta_rows = compute_deltas_vs_baseline(
        all_rows,
        baseline_run="baseline",                      # 如果你的 baseline run 名字不是 "baseline"，改这里
        compare_runs=("with","without"),
        metric_fields=("LL_canary_avg","LL_canary_sum"),
    )
    write_delta_csv(delta_rows, outdir / "delta_vs_baseline.csv")
    print(f"Saved per-sample deltas to: {outdir}/delta_vs_baseline.csv")

    # 2) 汇总 Δ(avg) 统计（按 run × bucket）
    delta_summary = summarize_deltas(delta_rows, metric="delta_LL_canary_avg")
    if delta_summary:
        out_csv = outdir / "delta_summary_LLavg.csv"
        write_summary_csv(delta_summary, out_csv)
        print("\n===== Δ SUMMARY (by run × bucket; metric=delta_LL_canary_avg) =====")
        for r in delta_summary:
            print(f"[{r['run']:^12}] [{r['bucket']:^7}] N={r['N']:4d} | "
                  f"mean={r['delta_LL_canary_avg_mean']:.4f} | p25={r['delta_LL_canary_avg_p25']:.4f} "
                  f"| median={r['delta_LL_canary_avg_median']:.4f} | p75={r['delta_LL_canary_avg_p75']:.4f}")
        print(f"\nSaved delta summary CSV to: {out_csv}")

    # 3) Δ(with-baseline) vs Δ(without-baseline) 的配对显著性（ALL + 按 bucket）
    print("\n===== PAIRED SIGNIFICANCE on Δ (avg) =====")
    paired_sig_on_deltas(delta_rows, run_a="with", run_b="without", bucket=None, metric="delta_LL_canary_avg")
    for b in ["seen","unseen","control"]:
        paired_sig_on_deltas(delta_rows, run_a="with", run_b="without", bucket=b, metric="delta_LL_canary_avg")


    print("\nDone.")

if __name__ == "__main__":
    main()

# python tools/batch_eval_canary_simple.py \
#   --model_id google/gemma-3-4b-it \
#   --run with=outputs/reinforce-simple-20250918-224643/step-250/adapter_model.safetensors \
#   --run without=outputs/reinforce-simple-20250918-220932/step-150/adapter_model.safetensors \
#   --run baseline=none \
#   --data data/infer/infer_seen.jsonl \
#   --data data/infer/infer_unseen.jsonl \
#   --data data/infer/infer_control.jsonl \
#   --outdir outputs/eval_out_4b \
#   --by_bucket_sig


# python tools/batch_eval_canary_simple.py \
#   --model_id google/gemma-3-270m-it \
#   --run with=.../step-250/adapter_model.safetensors \
#   --run without=.../step-100/adapter_model.safetensors \
#   --data data/infer_seen.jsonl \
#   --data data/infer_unseen.jsonl \
#   --data data/infer_control.jsonl \
#   --outdir outputs/eval_out_simple \
#   --reuse_results
