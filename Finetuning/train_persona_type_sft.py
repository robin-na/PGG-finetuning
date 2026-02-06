import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
import inspect


def build_text(tokenizer, messages, add_generation_prompt=False):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    # Fallback template
    parts = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def tokenize_example(example, tokenizer, max_length):
    messages = example["messages"]
    full_text = build_text(tokenizer, messages, add_generation_prompt=False)
    prompt_text = build_text(tokenizer, messages[:-1], add_generation_prompt=True)

    full = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )
    prompt = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    if prompt_len > 0:
        labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": prompt_len,
        "total_len": len(input_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name or path.")
    parser.add_argument("--train", required=True, help="Path to train JSONL.")
    parser.add_argument("--eval", default=None, help="Path to eval JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-response-tokens", type=int, default=1)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable evaluation during training even if --eval is provided.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory.",
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="",
        help="Comma-separated list of target module names for LoRA.",
    )
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_target_modules = None
    if args.lora_target_modules:
        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    if args.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("BitsAndBytesConfig is required for --use-qlora") from exc
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("peft is required for --use-qlora") from exc

        compute_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.bnb_4bit_compute_dtype]

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        device_map = "auto"
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device_map = None

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quant_config,
            device_map=device_map,
        )
        model = prepare_model_for_kbit_training(model)

        if lora_target_modules is None:
            model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
            if "qwen" in model_type:
                lora_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )

    if args.use_lora and not args.use_qlora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft is required for --use-lora") from exc
        if lora_target_modules is None:
            model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
            if "qwen" in model_type:
                lora_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_cfg)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable parameters found. LoRA target_modules likely did not match. "
            "Pass --lora-target-modules (e.g., q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj for Qwen)."
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.no_eval or not args.eval:
        data_files = {"train": args.train}
    else:
        data_files = {"train": args.train, "validation": args.eval}
    raw = load_dataset("json", data_files=data_files)

    def _tokenize(example):
        return tokenize_example(example, tokenizer, args.max_seq_len)

    tokenized = raw.map(_tokenize, remove_columns=raw["train"].column_names)

    def _keep(example):
        return (example["total_len"] - example["prompt_len"]) >= args.min_response_tokens

    tokenized = tokenized.filter(_keep)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    eval_dataset = None
    if not args.no_eval and "validation" in tokenized:
        eval_dataset = tokenized["validation"]

    eval_strategy_value = "no" if eval_dataset is None else "steps"
    ta_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "evaluation_strategy": eval_strategy_value,
        "eval_strategy": eval_strategy_value,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "seed": args.seed,
        "report_to": "none",
        "remove_unused_columns": False,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    valid = set(sig.parameters.keys())
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid}
    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
