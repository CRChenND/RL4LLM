import os, time, json, hashlib, random
from typing import List, Dict, Any, Optional
import requests

JUDGE_SYS_PROMPT = (
  "You are a strict, no-nonsense evaluator. "
  "Score the ASSISTANT_ANSWER only for factual correctness with respect to the REFERENCE_ANSWER. "
  "Ignore style and verbosity. Output ONLY a JSON object: {\"score\": float in [0,1], \"reason\": string}."
)

PAIRWISE_SYS_PROMPT = (
  "You are a strict evaluator. Given a question and a reference answer, score EACH candidate's factual correctness "
  "relative to the reference independently on [0,1]. Output ONLY JSON: {\"scores\": [float,...], \"reasons\": [string,...]} "
  "with the same length/order as candidates."
)

def _hash_key(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

class JudgeClient:
    def __init__(self, cfg: Dict[str, Any]):
        self.api_url = cfg["api_url"]
        self.model = cfg["model"]
        self.api_key = os.environ.get(cfg["api_key_env"], "")
        if not self.api_key:
            raise RuntimeError(f"Missing API key env: {cfg['api_key_env']}")
        self.timeout = cfg.get("timeout_s", 30)
        self.max_retries = cfg.get("max_retries", 3)
        self.cooldown = cfg.get("cooldown_s", 2)
        self.cache_path = cfg.get("cache_path", None)
        self._cache = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["value"]
                    except Exception:
                        pass

    def _cached(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    def _commit(self, key: str, value: Dict[str, Any]):
        self._cache[key] = value
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def _post(self, messages: List[Dict[str,str]]) -> str:
        hdr = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # Optional but recommended by OpenRouter:
            "HTTP-Referer": "https://example.com/rl-judge",
            "X-Title": "RL Judger",
        }
        body = {"model": self.model, "messages": messages, "temperature": 0}
        last_err = None
        for attempt in range(1, self.max_retries+1):
            try:
                r = requests.post(self.api_url, headers=hdr, json=body, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    return data["choices"][0]["message"]["content"]
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = str(e)
            time.sleep(self.cooldown * attempt)
        raise RuntimeError(f"Judge request failed after {self.max_retries} retries: {last_err}")

    # ----- public APIs -----

    def score_pointwise(self, question: str, ref: str, answer: str) -> Dict[str, Any]:
        payload = {
            "mode": "pointwise",
            "model": self.model,
            "question": question, "reference": ref, "answer": answer,
            "sys": JUDGE_SYS_PROMPT,
        }
        key = _hash_key(payload)
        if (v := self._cached(key)) is not None:
            return v
        messages = [
            {"role": "system", "content": JUDGE_SYS_PROMPT},
            {"role": "user", "content": (
                f"QUESTION:\n{question}\n\nREFERENCE_ANSWER:\n{ref}\n\nASSISTANT_ANSWER:\n{answer}\n\n"
                "Return JSON now."
            )},
        ]
        txt = self._post(messages)
        # be robust to minor deviations
        try:
            obj = json.loads(txt)
            score = float(obj.get("score", 0.0))
            reason = str(obj.get("reason", ""))
        except Exception:
            # fallback: look for a number
            import re
            m = re.search(r"[-+]?\d*\.?\d+", txt)
            score = float(m.group()) if m else 0.0
            reason = txt[:200]
        out = {"score": max(0.0, min(1.0, score)), "reason": reason}
        self._commit(key, out)
        return out

    def score_kwise(self, question: str, ref: str, candidates: List[str]) -> Dict[str, Any]:
        payload = {
            "mode": "kwise",
            "model": self.model,
            "question": question, "reference": ref, "candidates": candidates,
            "sys": PAIRWISE_SYS_PROMPT,
        }
        key = _hash_key(payload)
        if (v := self._cached(key)) is not None:
            return v
        cand_block = "\n\n".join([f"CANDIDATE {i+1}:\n{c}" for i,c in enumerate(candidates)])
        messages = [
            {"role": "system", "content": PAIRWISE_SYS_PROMPT},
            {"role": "user", "content": (
                f"QUESTION:\n{question}\n\nREFERENCE_ANSWER:\n{ref}\n\n{cand_block}\n\nReturn JSON now."
            )},
        ]
        txt = self._post(messages)
        try:
            obj = json.loads(txt)
            scores = [float(x) for x in obj.get("scores", [])]
            reasons = [str(x) for x in obj.get("reasons", [])]
        except Exception:
            # very defensive fallback
            scores = [0.0]*len(candidates)
            reasons = ["parse_error"]*len(candidates)
        scores = [max(0.0, min(1.0, s)) for s in scores]
        out = {"scores": scores, "reasons": reasons}
        self._commit(key, out)
        return out
