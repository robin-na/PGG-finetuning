from __future__ import annotations

from typing import Any, Optional, Tuple

from utils import log


def load_model(base_model: str, adapter_path: Optional[str], use_peft: bool) -> Tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
            log("[attn] SDPA flash enabled (flash=True, mem_efficient=True, math=False)")
        except Exception as e:  # pragma: no cover - hardware dependent
            log("[attn] Could not enable SDPA flash:", e)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device in ("cuda", "mps") else torch.float32)
    )

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    if device != "cpu":
        model = model.to(device)

    if use_peft and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if device != "cpu":
            model = model.to(device)

    model.eval()
    log(
        "[model] device=%s dtype=%s attn_impl=%s use_cache=%s peft=%s"
        % (
            model.device,
            dtype,
            getattr(model.config, "attn_implementation", None),
            getattr(model.config, "use_cache", None),
            use_peft and bool(adapter_path),
        )
    )
    if str(model.device).startswith("cpu"):
        log("[warn] model on CPU → expect slow inference")
    return tok, model
