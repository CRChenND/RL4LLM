import json
import re

def clean_text(text: str) -> str:
    # 保留：字母数字、常用标点、空格和换行
    return re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\-\n]", "", text)

input_path = "./data/qa_eval_new.jsonl"
output_path = "./data/qa_eval_clean.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        obj = json.loads(line)

        if "prompt" in obj:
            obj["prompt"] = clean_text(obj["prompt"])
        if "response" in obj:
            obj["response"] = clean_text(obj["response"])

        # 你要是希望 meta 里的内容也一起清理，可以加上：
        if "meta" in obj:
            obj["meta"] = {k: clean_text(str(v)) for k, v in obj["meta"].items()}

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Cleaned data written to {output_path}")