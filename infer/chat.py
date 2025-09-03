#!/usr/bin/env python3
import os
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel


def _auto_device() -> str:
    try:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_dtype(device: str, dtype: str):
    # Be conservative on CPU and MPS to avoid NaNs
    if device in ("cpu", "mps"):
        return torch.float32
    if dtype == "bfloat16" and hasattr(torch, "bfloat16"):
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _load_cfg(path: str) -> dict:
    cfg = yaml.safe_load(open(path))
    if isinstance(cfg, dict) and "inherit" in cfg:
        base = yaml.safe_load(open(cfg["inherit"]))
        base.update(cfg)
        cfg = base
    return cfg


def _format_chat(tok: AutoTokenizer, messages: list) -> str:
    # Prefer tokenizer's chat template if available
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback simple format
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
        else:
            parts.append(f"User: {content}\n")
    parts.append("Assistant: ")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Interactive chat with a base model + LoRA adapter.")
    ap.add_argument("--adapter", required=False, default=None, help="Path to adapter dir or adapter_model.safetensors file; omit to chat with base model only")
    ap.add_argument("--model_id", default=None, help="Base model ID (e.g., google/gemma-3-270m-it)")
    ap.add_argument("--config", default="configs/base.yaml", help="YAML config to read defaults (model_id)")
    ap.add_argument("--device", default="auto", help="Device: auto|cuda|mps|cpu")
    ap.add_argument("--dtype", default="float16", help="float16|bfloat16|float32 (cpu forces float32)")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max_input_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    adapter_dir = args.adapter
    if adapter_dir:
        if adapter_dir.endswith("adapter_model.safetensors"):
            adapter_dir = os.path.dirname(adapter_dir)
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Model ID: CLI > config
    model_id = args.model_id
    if model_id is None:
        cfg = _load_cfg(args.config)
        model_id = cfg.get("model_id")
        if not model_id:
            raise ValueError("--model_id not provided and model_id missing in config")

    # HF auth (for gated models)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    auth = {"token": hf_token} if hf_token else {}

    device = _auto_device() if args.device == "auto" else args.device
    dtype = _resolve_dtype(device, args.dtype)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **auth)
    if tok.pad_token_id is None and tok.eos_token is not None:
        try:
            tok.pad_token = tok.eos_token
        except Exception:
            pass

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map={"": device}, **auth
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base
    model.eval()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("Enter ':q' to quit, ':reset' to clear history.")
    while True:
        try:
            user = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in (":q", ":quit", ":exit"):
            break
        if user == ":reset":
            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("[history cleared]")
            continue
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        prompt_text = _format_chat(tok, messages)
        inputs = tok([prompt_text], return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length).to(device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                generation_config=GenerationConfig(pad_token_id=tok.eos_token_id),
                eos_token_id=tok.eos_token_id,
            )
        offs = inputs["input_ids"].shape[1]
        out = tok.decode(gen[0, offs:], skip_special_tokens=True).strip()
        print(f"Assistant> {out}")
        messages.append({"role": "assistant", "content": out})


if __name__ == "__main__":
    main()
