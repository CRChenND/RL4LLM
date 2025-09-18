# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, hashlib, time
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

EMPTY_BASELINE = 0.5710

def tlog(msg):
    print(f"[TLOG {time.strftime('%H:%M:%S')}] {msg}", flush=True)

def _hash16(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

class EmbCache:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.mem: Dict[str, List[float]] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        k, v = rec.get("k"), rec.get("v")
                        if isinstance(k, str) and isinstance(v, list):
                            self.mem[k] = v
                    except Exception:
                        pass

    def get(self, k: str):
        return self.mem.get(k)

    def put(self, k: str, v: List[float]):
        if k in self.mem: return
        self.mem[k] = v
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")

class SemanticReward:
    """Reward = semantic similarity between response and reference."""

    def __init__(self, cfg: Dict[str, Any]):
        if torch.cuda.is_available(): self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device = "mps"
        else: self.device = "cpu"

        assert SentenceTransformer is not None, "Please `pip install sentence-transformers`."
        self.model_id = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.model    = SentenceTransformer(self.model_id, device=self.device)
        self.cache    = EmbCache(cfg.get("cache_path"))

        self.encode_bs = int(cfg.get("encode_batch_size", 256))
        self.normalize_vec = bool(cfg.get("normalize_vec", True))
        self.rescale01     = bool(cfg.get("rescale01", True))
        self.gamma_sem     = float(cfg.get("shaping", {}).get("gamma_sem", 1.5))

        n0 = 0
        if self.cache and self.cache.path and os.path.exists(self.cache.path):
            n0 = sum(1 for _ in open(self.cache.path, "r", encoding="utf-8"))
        # tlog(f"SemanticReward ready. device={self.device} encode_bs={self.encode_bs} cache='{self.cache.path}' lines={n0}")


    def __call__(self, responses, references, **kw):
        return self.score(responses, references)

    def _embed_batch(self, texts: List[str], cache_refs: bool) -> torch.Tensor:
        # cache_refs=True 表示是 reference，允许读/写 cache；False 表示 response，不写 cache
        vecs, batch_texts, miss_keys, miss_idx = [None]*len(texts), [], [], []
        for i, t in enumerate(texts):
            s = (t or "").strip()
            k = _hash16(s)
            if cache_refs and self.cache:
                v = self.cache.get(k)
                if v is not None:
                    vecs[i] = torch.tensor(v, device=self.device, dtype=torch.float32)
                    continue
            batch_texts.append(s); miss_keys.append(k); miss_idx.append(i)

        if batch_texts:
            em = self.model.encode(batch_texts, convert_to_tensor=True,
                                device=self.device, normalize_embeddings=False)
            for k, i, v in zip(miss_keys, miss_idx, em):
                vecs[i] = v
                if cache_refs and self.cache:
                    self.cache.put(k, v.detach().cpu().tolist())

        emt = torch.stack(vecs, dim=0)
        if self.normalize_vec:
            emt = F.normalize(emt, p=2, dim=1)
        return emt

    def score(self, responses: List[str], references: List[str]) -> List[float]:
        assert len(responses) == len(references)
        R = self._embed_batch(responses, cache_refs=False)   # 不写 cache
        G = self._embed_batch(references, cache_refs=True)   # 参考写 cache
        sim = (R * G).sum(dim=1)  # 归一化后内积=cos
        if not self.rescale01:
            pass
        else:
            sim = 0.5 * (sim + 1.0)
        sim = (sim - EMPTY_BASELINE) / (1 - EMPTY_BASELINE)
        if self.gamma_sem != 1.0:
            sim = sim.clamp(0.0, 1.0).pow(self.gamma_sem)
        return [float(x) for x in sim.clamp(0.0, 1.0).detach().cpu()]

__all__ = ["SemanticReward"]
