from __future__ import annotations

from typing import Any, Optional, Tuple


def load_model(
    base_model: str,
    adapter_path: Optional[str],
    use_peft: bool = True,
    tokenizer_path: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device in ("cuda", "mps") else torch.float32)
    )

    tok_source = tokenizer_path or adapter_path or base_model
    try:
        tok = AutoTokenizer.from_pretrained(tok_source, use_fast=True, trust_remote_code=trust_remote_code)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=trust_remote_code)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )

    if device != "cpu":
        model = model.to(device)

    if use_peft and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if device != "cpu":
            model = model.to(device)

    model.eval()
    return tok, model
