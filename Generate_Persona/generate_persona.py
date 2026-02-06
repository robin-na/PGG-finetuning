from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from config_loader import TreatmentConfig, load_treatment_configs
from model_loader import load_model
from prompt_builder import build_chat_prompt, build_messages


def read_base_model_name(adapter_path: Path) -> Optional[str]:
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("base_model_name_or_path")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True))
            f.write("\n")


def _tokenize_prompts(tok: Any, prompts: List[str], device: torch.device) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    max_length = getattr(tok, "model_max_length", None)
    use_truncation = max_length is not None and max_length < 1_000_000
    tokenization_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": use_truncation,
    }
    if use_truncation:
        tokenization_kwargs["max_length"] = max_length

    inputs = tok(prompts, **tokenization_kwargs)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_lengths = [inputs["input_ids"][i].shape[-1] for i in range(len(prompts))]
    return inputs, prompt_lengths


@torch.inference_mode()
def _generate_chunk(
    tok: Any,
    model: Any,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> List[str]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    inputs, prompt_lengths = _tokenize_prompts(tok, prompts, model.device)
    gen_out = model.generate(
        **inputs,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )

    outputs: List[str] = []
    for i in range(len(prompts)):
        prompt_len = prompt_lengths[i]
        gen_ids = gen_out[i][prompt_len:]
        new_text = tok.decode(gen_ids, skip_special_tokens=True)
        outputs.append(new_text)
    return outputs


def _clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def clean_generation(text: str) -> str:
    if not text:
        return text
    s = text.strip()
    if "<think>" in s and "</think>" in s:
        start = s.find("<think>")
        end = s.find("</think>")
        if end > start:
            s = s[end + len("</think>") :].strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            if lines[-1].startswith("```"):
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            s = "\n".join(lines).strip()
    if not (s.startswith("{") and s.endswith("}")):
        left = s.find("{")
        right = s.rfind("}")
        if left != -1 and right != -1 and right > left:
            s = s[left : right + 1].strip()
    return s


def generate_with_oom_fallback_stream(
    tok: Any,
    model: Any,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> Iterable[Tuple[int, List[str]]]:
    if not prompts:
        return

    chunk_size = batch_size if batch_size and batch_size > 0 else len(prompts)
    env_chunk_size = os.getenv("PGG_LOCAL_GENERATION_CHUNK_SIZE")
    if env_chunk_size:
        try:
            env_value = int(env_chunk_size)
        except ValueError:
            env_value = None
        if env_value and env_value > 0:
            chunk_size = min(chunk_size, env_value)

    idx = 0
    while idx < len(prompts):
        try:
            chunk_prompts = prompts[idx : idx + chunk_size]
            chunk_out = _generate_chunk(
                tok,
                model,
                chunk_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            yield idx, chunk_out
            idx += len(chunk_prompts)
        except torch.OutOfMemoryError as exc:
            _clear_memory()
            if chunk_size <= 1:
                raise exc
            chunk_size = max(1, chunk_size // 2)
            print(
                f"[warn] CUDA OOM during generation with batch size; retrying with reduced batch size {chunk_size}.",
                flush=True,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            _clear_memory()
            if chunk_size <= 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            print(
                f"[warn] OOM during generation with batch size; retrying with reduced batch size {chunk_size}.",
                flush=True,
            )


def _ensure_output_dir(base_output: Path, base_model: str) -> Path:
    model_path = Path(base_model)
    if model_path.is_absolute():
        model_path = Path(*model_path.parts[1:]) if len(model_path.parts) > 1 else Path("absolute_model")
    return base_output / model_path


def _build_prompts(
    tok: Any,
    records: List[TreatmentConfig],
    enable_thinking: Optional[bool],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    prompts: List[str] = []
    input_rows: List[Dict[str, Any]] = []
    for rec in records:
        messages = build_messages(rec.config, rec.n_players)
        prompt_text = build_chat_prompt(
            tok,
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(prompt_text)
        input_rows.append(
            {
                "CONFIG_treatmentName": rec.treatment_name,
                "prompt": prompt_text,
            }
        )
    return prompts, input_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-path",
        default="Finetuning/out_persona_types_lora",
        help="Path to LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name or path. If omitted, read from adapter_config.json.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path or name. Defaults to adapter path if available.",
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed_data/df_analysis_val.csv",
        help="CSV with CONFIG_* columns.",
    )
    parser.add_argument(
        "--treatment-col",
        default="CONFIG_treatmentName",
        help="Column to deduplicate on and use as key.",
    )
    parser.add_argument(
        "--output-root",
        default="Persona/LLM_mapped",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Initial batch size; will be reduced on OOM.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow model reasoning tokens (e.g., <think> blocks) in outputs.",
    )
    parser.add_argument(
        "--no-clean-json",
        action="store_true",
        help="Disable postprocessing that strips reasoning and extracts JSON.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Set trust_remote_code=True when loading models/tokenizer.",
    )

    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    base_model = args.base_model or read_base_model_name(adapter_path)
    if not base_model:
        raise ValueError(
            "Base model not provided and could not be inferred from adapter_config.json. "
            "Set --base-model explicitly."
        )

    tokenizer_path = args.tokenizer_path or str(adapter_path)

    records, stats = load_treatment_configs(Path(args.input_csv), args.treatment_col)
    print(
        f"[data] total_rows={stats['total_rows']} dedup_rows={stats['dedup_rows']} "
        f"dropped={stats['dropped_rows']} kept={stats['kept']}",
        flush=True,
    )
    if not records:
        print("[data] no records to process; exiting.", flush=True)
        return

    tok, model = load_model(
        base_model=base_model,
        adapter_path=str(adapter_path),
        use_peft=True,
        tokenizer_path=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )

    prompts, input_rows = _build_prompts(tok, records, enable_thinking=args.enable_thinking)

    output_dir = _ensure_output_dir(Path(args.output_root), base_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = output_dir / "persona_type_inputs.jsonl"
    outputs_path = output_dir / "persona_type_outputs.jsonl"

    write_jsonl(inputs_path, input_rows)

    with open(outputs_path, "w", encoding="utf-8") as out_f:
        for start_idx, gen_texts in generate_with_oom_fallback_stream(
            tok,
            model,
            prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        ):
            for offset, gen_text in enumerate(gen_texts):
                rec = records[start_idx + offset]
                prompt_text = prompts[start_idx + offset]
                cleaned = gen_text if args.no_clean_json else clean_generation(gen_text)
                row = {
                    "CONFIG_treatmentName": rec.treatment_name,
                    "prompt": prompt_text,
                    "generation": cleaned,
                }
                out_f.write(json.dumps(row, ensure_ascii=True))
                out_f.write("\n")
            out_f.flush()

    print(f"[output] inputs: {inputs_path}", flush=True)
    print(f"[output] outputs: {outputs_path}", flush=True)


if __name__ == "__main__":
    main()
