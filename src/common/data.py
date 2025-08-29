import json, random, itertools
from typing import List, Tuple, Dict, Optional

def load_qa_jsonl(path:str, max_samples:Optional[int]=None) -> List[Dict]:
    items=[]
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            obj=json.loads(line)
            # expect keys: prompt, response
            if "prompt" in obj and "response" in obj:
                items.append({"q": obj["prompt"], "a": obj["response"]})
    if max_samples:
        items = items[:max_samples]
    return items

def format_prompt(q:str, template:str) -> str:
    return template.replace("{question}", q)

def qa_batch(
    dataset: List[Dict],
    batch_size:int,
    template:str
) -> Tuple[List[str], List[str]]:
    batch = random.sample(dataset, k=batch_size)
    prompts = [format_prompt(x["q"], template) for x in batch]
    refs    = [x["a"] for x in batch]
    return prompts, refs
