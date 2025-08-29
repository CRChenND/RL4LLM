import os, json, hashlib, pathlib
import torch, torch.nn.functional as F
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

class EmbCache:
    def __init__(self, path: str|None):
        self.path = path
        self.mem: Dict[str, list[float]] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.mem[rec["k"]] = rec["v"]
                    except Exception:
                        pass

    def get(self, k: str): return self.mem.get(k)
    def put(self, k: str, v: list[float]):
        self.mem[k] = v
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")

class SemanticReward:
    def __init__(self, cfg: Dict[str,Any]):
        dev = "mps" if cfg.get("device","auto")=="auto" and torch.backends.mps.is_available() else cfg.get("device","cpu")
        self.device = dev
        self.model = SentenceTransformer(cfg["model"], device=self.device)
        self.normalize = bool(cfg.get("normalize", True))
        self.rescale01 = bool(cfg.get("rescale01", True))
        self.length_clip = int(cfg.get("length_clip", 512))
        self.cache = EmbCache(cfg.get("cache_path"))

    def _prep(self, s: str) -> str:
        if self.length_clip and len(s) > self.length_clip:
            return s[:self.length_clip]
        return s

    def _embed(self, texts: List[str]) -> torch.Tensor:
        # cache-aware embedding
        batch, misses, miss_idx = [], [], []
        vecs = [None]*len(texts)
        for i,t in enumerate(texts):
            t = self._prep(t)
            k = _hash(t)
            if self.cache and (v := self.cache.get(k)) is not None:
                vecs[i] = torch.tensor(v, device=self.device, dtype=torch.float32)
            else:
                batch.append(t); misses.append(k); miss_idx.append(i)
        if batch:
            em = self.model.encode(batch, convert_to_tensor=True, device=self.device, normalize_embeddings=False)
            for k, i, v in zip(misses, miss_idx, em):
                if self.normalize:
                    v = F.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                vecs[i] = v
                if self.cache:
                    self.cache.put(k, v.detach().cpu().tolist())
        # stack
        emt = torch.stack(vecs, dim=0)
        if self.normalize:
            emt = F.normalize(emt, p=2, dim=1)
        return emt

    def score(self, responses: List[str], references: List[str]) -> List[float]:
        assert len(responses)==len(references)
        R = self._embed(responses)
        G = self._embed(references)
        cos = (R*G).sum(dim=1)  # cosine if normalized
        if not self.normalize:
            cos = F.cosine_similarity(R, G, dim=1)
        if self.rescale01:
            cos = 0.5*(cos + 1.0)
        # clamp numeric noise
        cos = torch.clamp(cos, 0.0, 1.0)
        return [float(x) for x in cos.detach().cpu()]
