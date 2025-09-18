# scripts/reward_anchors.py
import os, sys, json, argparse, random, datetime, yaml
from typing import List, Dict, Any, Tuple

# 让脚本能从项目根导入 src.*
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(CURR_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src.common.data import load_qa_jsonl
from src.common.reward_semantic import SemanticReward

def load_cfg(path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
    # 兼容 base.yaml 继承
    if "inherit" in cfg:
        base = yaml.safe_load(open(cfg["inherit"], "r", encoding="utf-8"))
        base.update(cfg)
        cfg = base
    return cfg

# 数据集中问题/答案字段名可能不同，这里做兼容
CAND_Q_KEYS = ["question", "prompt", "input", "query", "q"]
CAND_A_KEYS = ["answer", "reference", "output", "label", "gold", "a"]

def _pick_first(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        if k in d and isinstance(d[k], str):
            return d[k]
    # 兜底：把非字符串也转成字符串（尽量别报错）
    for k in keys:
        if k in d:
            return str(d[k])
    return ""

def extract_qas(items: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    prompts, refs = [], []
    for it in items:
        q = _pick_first(it, CAND_Q_KEYS)
        a = _pick_first(it, CAND_A_KEYS)
        prompts.append(q or "")
        refs.append(a or "")
    return prompts, refs

def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))

def benchmark_reward_anchors(cfg_path: str,
                             max_n: int = 1000,
                             use_eval: bool = True,
                             first_line_only: bool = False) -> Dict[str, Any]:
    cfg = load_cfg(cfg_path)
    dscfg = cfg["dataset"]

    data_path = dscfg.get("eval_path") if use_eval and dscfg.get("eval_path") else dscfg["path"]
    data = load_qa_jsonl(data_path, max_n)
    prompts, refs = extract_qas(data)

    # 构建 reward（不改你训练用的 cfg，只在评测时可选覆盖 first_line_only）
    sr_cfg = dict(cfg.get("semantic_reward", {}))
    if first_line_only:
        sr_cfg = dict(sr_cfg)
        sr_cfg["first_line_only"] = True
    reward = SemanticReward(sr_cfg)

    # 1) 空回答（理论下界）
    empty = reward.score([""] * len(refs), refs)

    # 2) 打乱配对（“随机相关度”基线）
    refs_shuf = refs[:]
    random.seed(1234)
    random.shuffle(refs_shuf)
    shuffled = reward.score(refs_shuf, refs)

    # 3) 复述问题（检验“抄题惩罚”强度）
    copyq = reward.score(prompts, refs)

    # 4) 参考当作预测（上界；仍可能受长度/抄题惩罚略压）
    ref_as_pred = reward.score(refs, refs)

    # 也可加一个“短空白客套话”基线（可选）
    short_filler_resp = ["i don't know" for _ in refs]
    filler = reward.score(short_filler_resp, refs)

    out = {
        "n": len(refs),
        "dataset_path": data_path,
        "config": cfg_path,
        "anchors": {
            "empty": mean(empty),
            "shuffled": mean(shuffled),
            "copy_question": mean(copyq),
            "ref_as_pred": mean(ref_as_pred),
            "filler_idk": mean(filler),
        },
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reinforce.yaml")
    ap.add_argument("--max-n", type=int, default=1000)
    ap.add_argument("--eval", action="store_true", default=True,
                    help="优先使用 eval_path；若不存在则回落到 train")
    ap.add_argument("--no-eval", dest="eval", action="store_false")
    ap.add_argument("--first-line-only", action="store_true", default=False,
                    help="仅在评测时按首句/首行记分（不改训练配置）")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    res = benchmark_reward_anchors(
        cfg_path=args.config,
        max_n=args.max_n,
        use_eval=args.eval,
        first_line_only=args.first_line_only
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.outdir, exist_ok=True)
    out_json = os.path.join(args.outdir, f"reward_anchors_{ts}.json")

    print("\n=== Reward Anchors (标定上下限) ===")
    print(f"Samples: {res['n']} | dataset: {res['dataset_path']}")
    for k, v in res["anchors"].items():
        print(f"{k:>16}: {v:.4f}")
    print(f"\nSaved: {out_json}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
