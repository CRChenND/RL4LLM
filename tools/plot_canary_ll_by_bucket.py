#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KernelDensity

# ---------- helpers ----------
def ll_to_ppl(ll_avg):  # per-token perplexity
    return np.exp(-ll_avg)

def _bw_scott(x):
    # Scott's rule: h = sigma * n^(-1/5)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 1.0
    sigma = np.std(x, ddof=1) or 1.0
    return sigma * (n ** (-1/5))

def _bw_silverman(x):
    # Silverman's rule
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return 1.0
    sigma = np.std(x, ddof=1) or 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    a = min(sigma, iqr / 1.349) or 1.0
    return 0.9 * a * (n ** (-1/5))

def kde(ax, x, label=None, alpha=0.25, bw="scott", grids=512, pclip=0.005):
    """
    高斯KDE平滑分布曲线。
    bw: "scott" | "silverman" | float  手动带宽
    grids: 评估网格点数
    pclip: 两端分位裁剪，避免极端值影响
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return

    if isinstance(bw, str):
        if bw.lower() == "scott":
            h = _bw_scott(x)
        elif bw.lower() == "silverman":
            h = _bw_silverman(x)
        else:
            raise ValueError(f"Unknown bw spec: {bw}")
    else:
        h = float(bw)

    lo = np.quantile(x, pclip)
    hi = np.quantile(x, 1 - pclip)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.min(x), np.max(x)
        if lo == hi:
            return
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    grid = np.linspace(lo - pad, hi + pad, int(grids))

    kde_est = KernelDensity(kernel="gaussian", bandwidth=max(h, 1e-9))
    kde_est.fit(x.reshape(-1, 1))
    log_dens = kde_est.score_samples(grid.reshape(-1, 1))
    dens = np.exp(log_dens)

    ax.plot(grid, dens, label=label)
    ax.fill_between(grid, dens, alpha=alpha)

def prepare_score(df_bucket, metric: str):
    """
    Returns (scores, labels) for ROC.
    labels: with=1, without=0
    scores: higher -> more likely from WITH
    """
    d = df_bucket.copy()
    d = d[d["run"].isin(["with", "without"])]

    if metric == "ppl":  # lower perplexity = higher score for WITH; flip sign so higher=WITH
        s = -ll_to_ppl(d["LL_canary_avg"].to_numpy())
    elif metric == "LL_canary_avg":
        s = d["LL_canary_avg"].to_numpy()
    elif metric == "LL_canary_sum":
        s = d["LL_canary_sum"].to_numpy()
    else:
        # fallback: any numeric column; assume higher=better
        s = d[metric].to_numpy()

    y = (d["run"] == "with").astype(int).to_numpy()
    return s, y

# ---------- main plotting ----------
def make_plots(scores_csv: str, outdir: str, metric: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(scores_csv)

    # sanity: keep the columns we need
    needed = {"run", "bucket"}
    if metric == "ppl":
        needed |= {"LL_canary_avg"}
    else:
        needed |= {metric}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    for bucket in ["seen", "unseen"]:
        dB = df[df["bucket"] == bucket].copy()
        if dB.empty:
            print(f"[WARN] no rows for bucket={bucket}")
            continue

        # ---- Distribution: WITH vs WITHOUT on the chosen metric ----
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        if metric == "ppl":
            x_with = ll_to_ppl(dB.loc[dB.run == "with", "LL_canary_avg"])
            x_without = ll_to_ppl(dB.loc[dB.run == "without", "LL_canary_avg"])
            xlabel = "Per-token Perplexity (lower is more confident)"
        else:
            x_with = dB.loc[dB.run == "with", metric]
            x_without = dB.loc[dB.run == "without", metric]
            xlabel = f"{metric} (higher is more confident)"

        ax.hist(x_without, bins=40, density=True, alpha=0.3, label="without")
        kde(ax, x_without, label="without", alpha=0.25,
            bw=args.kde_bw, grids=args.kde_grids, pclip=args.kde_pclip)
        ax.hist(x_with, bins=40, density=True, alpha=0.3, label="with")
        kde(ax, x_with, label="with", alpha=0.25,
            bw=args.kde_bw, grids=args.kde_grids, pclip=args.kde_pclip)
        ax.set_title(f"{metric} distribution — with vs without (bucket={bucket})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(Path(outdir) / f"dist_{metric}_{bucket}.png"), dpi=160)
        plt.close(fig)

        # ---- ROC: WITH (1) vs WITHOUT (0) within this bucket ----
        s, y = prepare_score(dB, metric)
        # Must have both classes
        if len(np.unique(y)) < 2:
            print(f"[WARN] ROC skipped for bucket={bucket}: only one class present.")
            continue

        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_title(f"ROC — detect WITH vs WITHOUT (bucket={bucket}, metric={metric})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(str(Path(outdir) / f"roc_with_vs_without_{metric}_{bucket}.png"), dpi=160)
        plt.close(fig)

    print(f"Saved plots to: {outdir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True, help="Path to canary_scores.csv")
    ap.add_argument("--outdir", default="viz_canary", help="Output directory for PNGs")
    ap.add_argument("--metric", default="LL_canary_avg",
                    help="One of: LL_canary_avg | LL_canary_sum | ppl | <any numeric col>")
    ap.add_argument("--kde_bw", default="scott", help="KDE bandwidth: scott|silverman|<float>")
    ap.add_argument("--kde_grids", type=int, default=128, help="KDE evaluation points")
    ap.add_argument("--kde_pclip", type=float, default=0.005, help="Tail clipping quantile")

    args = ap.parse_args()
    make_plots(args.scores_csv, args.outdir, args.metric)

# python tools/plot_canary_ll_by_bucket.py \
#   --scores_csv outputs/eval_out_simple/canary_scores.csv \
#   --metric LL_canary_avg \
#   --outdir outputs/eval_out_simple

