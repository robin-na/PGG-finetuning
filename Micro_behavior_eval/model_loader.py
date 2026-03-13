from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    from .utils import log
except ImportError:
    from utils import log


def _resolve_compute_dtype(dtype_name: str, fallback_dtype: Any) -> Any:
    import torch

    normalized = str(dtype_name or "auto").strip().lower()
    mapping = {
        "auto": fallback_dtype,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported quant compute dtype '{dtype_name}'. Use auto, bf16, fp16, or fp32.")
    return mapping[normalized]


def load_model(
    base_model: str,
    adapter_path: Optional[str],
    use_peft: bool,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    quant_compute_dtype: str = "auto",
) -> Tuple[Any, Any]:
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

    if load_in_8bit and load_in_4bit:
        raise ValueError("Choose at most one quantization mode: load_in_8bit or load_in_4bit.")

    quant_mode = None
    quant_dtype = None
    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if load_in_8bit or load_in_4bit:
        if device != "cuda":
            raise ValueError("bitsandbytes quantization in this pipeline requires CUDA.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError(
                "Quantized local loading requested but BitsAndBytesConfig is unavailable. "
                "Install bitsandbytes and a compatible transformers build."
            ) from exc

        quant_dtype = _resolve_compute_dtype(quant_compute_dtype, dtype)
        quant_mode = "8bit" if load_in_8bit else "4bit"
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=quant_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = quantization_config

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            attn_implementation="sdpa",
            **model_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs,
        )

    if quant_mode is None and device != "cpu":
        model = model.to(device)

    if use_peft and adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if quant_mode is None and device != "cpu":
            model = model.to(device)

    model.eval()
    device_label = str(getattr(model, "device", "unknown"))
    if getattr(model, "hf_device_map", None):
        device_label = f"hf_device_map={getattr(model, 'hf_device_map')}"
    log(
        "[model] device=%s dtype=%s attn_impl=%s use_cache=%s peft=%s quant=%s quant_compute_dtype=%s"
        % (
            device_label,
            dtype,
            getattr(model.config, "attn_implementation", None),
            getattr(model.config, "use_cache", None),
            use_peft and bool(adapter_path),
            quant_mode or "none",
            quant_dtype or "n/a",
        )
    )
    if str(device_label).startswith("cpu"):
        log("[warn] model on CPU -> expect slow inference")
    return tok, model
