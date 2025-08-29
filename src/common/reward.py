import json, re

def reward_exact_8(responses:list[str])->list[float]:
    out=[]
    for r in responses:
        first = r.strip().split()[0].strip(".,")
        out.append(1.0 if first=="8" else 0.0)
    return out

def reward_json_schema(responses, required_keys=("tool","args")):
    scores=[]
    for r in responses:
        try:
            obj=json.loads(r)
            ok = all(k in obj for k in required_keys)
            scores.append(1.0 if ok else 0.0)
        except Exception:
            scores.append(0.0)
    return scores
