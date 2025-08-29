import os, time

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def run_dir(root="outputs", tag="run"):
    ts=time.strftime("%Y%m%d-%H%M%S")
    path=f"{root}/{tag}-{ts}"
    ensure_dir(path); return path

def save_adapter(model, path):
    model.save_pretrained(path)
