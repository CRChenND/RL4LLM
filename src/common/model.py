import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model

def load_model_and_tokenizer(model_id: str, device: str, dtype: str, lora_cfg: dict):
    # --- auth (optional) ---
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    auth_kwargs = {"token": hf_token} if hf_token else {}

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **auth_kwargs)

    # Prefer PAD=EOS if missing (Gemma-style safe default)
    added_pad = False
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            # very rare: add a brand-new pad token and remember to resize embeddings
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad = True

    # --- dtype resolution: use fp32 on MPS for stability ---
    def _resolve_dtype(dev: str, req: str):
        if torch.backends.mps.is_available() and dev == "mps":
            return torch.float32          # <- key change for stability
        if dev == "cpu":
            return torch.float32
        if req == "bfloat16" and hasattr(torch, "bfloat16"):
            return torch.bfloat16
        if req == "float16":
            return torch.float16
        return torch.float32

    torch_dtype = _resolve_dtype(device, dtype)

    # --- base model ---
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map={"": device},
        **auth_kwargs
    )

    # if we added a new PAD token, resize embeddings to avoid index errors
    if added_pad:
        base.resize_token_embeddings(len(tok))

    # Mirror PAD/EOS on model config (guards sampling code)
    if getattr(base.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        base.config.pad_token_id = tok.pad_token_id
    if getattr(base.config, "eos_token_id", None) is None and tok.eos_token_id is not None:
        base.config.eos_token_id = tok.eos_token_id

    # --- LoRA ---
    lora = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(base, lora)

    # Set a safe generation_config used by your sampling helper
    model.generation_config = GenerationConfig(
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )

    # Eval mode by default; your training loop will switch to train() only for backward
    model.eval()

    return model, tok
