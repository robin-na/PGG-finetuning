from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_persona_summary_batch_inputs import (
    ALL_MODELS,
    ALL_SPLITS,
    MODEL_SLUGS,
    REPO_ROOT,
    _apply_us_country_filter,
    _batch_entry,
    _build_batched_twin_persona_summary_prompt,
    _build_context_groups,
    _estimate_input_tokens_heuristic,
    _load_simbench_rows,
    _load_twin_pids,
    _sample_twin_pids,
    _sanitize_token,
    _seed_from_parts,
    _simbench_csv_path,
    _twin_profiles_path,
    _write_csv,
    _write_jsonl,
)
from persona_summary_reconstruction import ALL_VARIANTS, PersonaSummaryReconstructor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECON_OUTPUT_DIR = SCRIPT_DIR / "ablation" / "output" / "persona_summary_reconstruction"


def _load_variant_summary_map(path: Path) -> dict[str, str]:
    summary_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = str(row.get("pid") or "").strip()
            persona_summary = str(row.get("persona_summary") or "").strip()
            if pid and persona_summary:
                summary_map[pid] = persona_summary
    if not summary_map:
        raise ValueError(f"No persona summaries loaded from {path}")
    return summary_map


def _default_run_name(
    *,
    split: str,
    model: str,
    summary_variant: str,
    num_samples_per_context: int,
    dataset_names: list[str],
    only_us_country_context: bool,
) -> str:
    name_parts = [
        _sanitize_token(split),
        _sanitize_token(summary_variant) + "_batched_seed_0",
        f"n{num_samples_per_context}",
        MODEL_SLUGS[model],
    ]
    if dataset_names:
        name_parts.append("datasets_" + "_".join(_sanitize_token(name) for name in dataset_names))
    if only_us_country_context:
        name_parts.append("us_only")
    return "__".join(name_parts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SimBench batch inputs using reconstructed Twin persona_summary variants."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--forecasting-root", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--split", type=str, choices=ALL_SPLITS, default="SimBenchPop")
    parser.add_argument("--model", type=str, choices=ALL_MODELS, default="gpt-5-mini")
    parser.add_argument("--summary-variant", type=str, choices=ALL_VARIANTS, required=True)
    parser.add_argument("--num-samples-per-context", type=int, default=64)
    parser.add_argument("--dataset-names", type=str, default="")
    parser.add_argument("--only-us-country-context", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_samples_per_context <= 0:
        raise ValueError("--num-samples-per-context must be positive.")

    recon_output_dir = DEFAULT_RECON_OUTPUT_DIR
    variant_jsonl = recon_output_dir / f"{args.summary_variant}.jsonl"
    if not variant_jsonl.exists():
        PersonaSummaryReconstructor(repo_root=args.repo_root).write_artifacts(recon_output_dir)
    summary_map = _load_variant_summary_map(variant_jsonl)

    dataset_names = [item.strip() for item in args.dataset_names.split(",") if item.strip()]
    simbench_rows = _load_simbench_rows(_simbench_csv_path(args.repo_root, args.split), args.split)
    if dataset_names:
        allowed = set(dataset_names)
        simbench_rows = [row for row in simbench_rows if row["dataset_name"] in allowed]
    if args.only_us_country_context:
        simbench_rows = _apply_us_country_filter(simbench_rows)
    simbench_rows = sorted(simbench_rows, key=lambda row: (row["dataset_name"], row["simbench_row_id"]))
    if not simbench_rows:
        raise ValueError("No SimBench rows selected after filtering.")

    twin_pids = [pid for pid in _load_twin_pids(_twin_profiles_path(args.repo_root)) if pid in summary_map]
    if not twin_pids:
        raise ValueError("No Twin pids overlap between local profiles and reconstructed summary variant.")

    batch_input_dir = args.forecasting_root / "batch_input"
    batch_output_dir = args.forecasting_root / "batch_output"
    metadata_root = args.forecasting_root / "metadata"
    results_root = args.forecasting_root / "results"
    for directory in [batch_input_dir, batch_output_dir, metadata_root, results_root]:
        directory.mkdir(parents=True, exist_ok=True)

    context_groups = _build_context_groups(simbench_rows)
    gold_rows = [
        {
            "simbench_row_id": row["simbench_row_id"],
            "simbench_split": row["simbench_split"],
            "dataset_name": row["dataset_name"],
            "option_labels": row["option_labels"],
            "gold_distribution": row["gold_distribution"],
            "group_size": int(row["group_size"]),
        }
        for row in simbench_rows
    ]

    run_name = args.run_name.strip() or _default_run_name(
        split=args.split,
        model=args.model,
        summary_variant=args.summary_variant,
        num_samples_per_context=args.num_samples_per_context,
        dataset_names=dataset_names,
        only_us_country_context=args.only_us_country_context,
    )
    batch_path = batch_input_dir / f"{run_name}.jsonl"
    metadata_dir = metadata_root / run_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = metadata_dir / "sample_prompts"
    sample_dir.mkdir(parents=True, exist_ok=True)

    request_manifest_path = metadata_dir / "request_manifest.csv"
    token_csv_path = metadata_dir / "request_token_estimates.csv"
    gold_targets_path = metadata_dir / "gold_targets.jsonl"
    selected_rows_path = metadata_dir / "selected_rows.csv"
    selected_contexts_path = metadata_dir / "selected_contexts.csv"
    shared_twin_panel_path = metadata_dir / "shared_twin_panel.json"

    _write_jsonl(gold_targets_path, gold_rows)
    _write_csv(
        selected_rows_path,
        [
            "simbench_row_id",
            "simbench_split",
            "dataset_name",
            "source_row_index",
            "group_size",
            "wave",
            "group_prompt_template",
            "group_prompt_variable_map_json",
            "rendered_group_prompt",
            "input_template",
            "question_body",
            "question_payload",
            "option_text_map_json",
            "option_labels_json",
            "gold_distribution_json",
        ],
        [
            {
                "simbench_row_id": row["simbench_row_id"],
                "simbench_split": row["simbench_split"],
                "dataset_name": row["dataset_name"],
                "source_row_index": row["source_row_index"],
                "group_size": row["group_size"],
                "wave": row["wave"],
                "group_prompt_template": row["group_prompt_template"],
                "group_prompt_variable_map_json": json.dumps(row["group_prompt_variable_map"], ensure_ascii=False),
                "rendered_group_prompt": row["rendered_group_prompt"],
                "input_template": row["input_template"],
                "question_body": row["question_body"],
                "question_payload": row["question_payload"],
                "option_text_map_json": json.dumps(row["option_text_map"], ensure_ascii=False),
                "option_labels_json": json.dumps(row["option_labels"], ensure_ascii=False),
                "gold_distribution_json": json.dumps(row["gold_distribution"], ensure_ascii=False),
            }
            for row in simbench_rows
        ],
    )
    _write_csv(
        selected_contexts_path,
        [
            "context_id",
            "simbench_split",
            "dataset_name",
            "rendered_group_prompt",
            "question_count",
            "dataset_task_note",
            "scale_definitions_json",
            "question_manifest_json",
        ],
        [
            {
                "context_id": context_group["context_id"],
                "simbench_split": context_group["simbench_split"],
                "dataset_name": context_group["dataset_name"],
                "rendered_group_prompt": context_group["rendered_group_prompt"],
                "question_count": len(context_group["question_manifest"]),
                "dataset_task_note": context_group.get("dataset_task_note") or "",
                "scale_definitions_json": json.dumps(context_group["scale_definitions"], ensure_ascii=False),
                "question_manifest_json": json.dumps(context_group["question_manifest"], ensure_ascii=False),
            }
            for context_group in context_groups
        ],
    )

    manifest_fieldnames = [
        "custom_id",
        "run_name",
        "model",
        "summary_variant",
        "simbench_split",
        "context_id",
        "dataset_name",
        "question_count",
        "sample_index",
        "twin_pid",
        "response_schema",
        "prompt_level",
        "question_manifest_json",
    ]
    derived_seed = _seed_from_parts(args.seed, args.split, args.summary_variant, "global_twin_panel")
    shared_twin_pids = _sample_twin_pids(
        twin_pids=twin_pids,
        num_samples=args.num_samples_per_context,
        seed_value=derived_seed,
    )
    shared_twin_panel_path.write_text(
        json.dumps(
            {
                "sampling_scope": "global_panel_across_contexts",
                "seed": int(args.seed),
                "derived_seed": int(derived_seed),
                "summary_variant": args.summary_variant,
                "simbench_split": args.split,
                "num_samples": int(args.num_samples_per_context),
                "twin_pids": shared_twin_pids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sample_written = False
    total_input_tokens = 0
    token_count = 0
    min_tokens: int | None = None
    max_tokens: int | None = None

    with (
        batch_path.open("w", encoding="utf-8") as batch_handle,
        request_manifest_path.open("w", encoding="utf-8", newline="") as manifest_handle,
        token_csv_path.open("w", encoding="utf-8", newline="") as token_handle,
    ):
        manifest_writer = csv.DictWriter(manifest_handle, fieldnames=manifest_fieldnames)
        manifest_writer.writeheader()
        token_writer = csv.DictWriter(token_handle, fieldnames=["custom_id", "input_token_estimate"])
        token_writer.writeheader()

        for context_group in context_groups:
            for sample_index, twin_pid in enumerate(shared_twin_pids, start=1):
                system_prompt, user_prompt = _build_batched_twin_persona_summary_prompt(
                    context_group,
                    summary_map[twin_pid],
                )
                custom_id = (
                    f"{_sanitize_token(args.split)}__{context_group['context_id']}__"
                    f"{_sanitize_token(args.summary_variant)}__s{sample_index:04d}"
                )
                batch_row = _batch_entry(
                    custom_id=custom_id,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                batch_handle.write(json.dumps(batch_row, ensure_ascii=False) + "\n")
                manifest_writer.writerow(
                    {
                        "custom_id": custom_id,
                        "run_name": run_name,
                        "model": args.model,
                        "summary_variant": args.summary_variant,
                        "simbench_split": args.split,
                        "context_id": context_group["context_id"],
                        "dataset_name": context_group["dataset_name"],
                        "question_count": len(context_group["question_manifest"]),
                        "sample_index": sample_index,
                        "twin_pid": twin_pid,
                        "response_schema": "batched_explanation_plus_answers",
                        "prompt_level": "batched_context_group",
                        "question_manifest_json": json.dumps(context_group["question_manifest"], ensure_ascii=False),
                    }
                )
                input_tokens = _estimate_input_tokens_heuristic(batch_row["body"]["messages"])
                token_writer.writerow({"custom_id": custom_id, "input_token_estimate": input_tokens})
                total_input_tokens += input_tokens
                token_count += 1
                min_tokens = input_tokens if min_tokens is None else min(min_tokens, input_tokens)
                max_tokens = input_tokens if max_tokens is None else max(max_tokens, input_tokens)
                if not sample_written:
                    sample_text = (
                        f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}\n\n"
                        f"[custom_id]\n{custom_id}\n"
                    )
                    (sample_dir / "sample_prompt.txt").write_text(sample_text, encoding="utf-8")
                    sample_written = True

    mean_tokens = (total_input_tokens / token_count) if token_count else 0.0
    (metadata_dir / "request_token_estimates.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "model": args.model,
                "summary_variant": args.summary_variant,
                "request_count": token_count,
                "total_input_tokens_estimate": total_input_tokens,
                "mean_input_tokens_estimate": mean_tokens,
                "min_input_tokens_estimate": min_tokens,
                "max_input_tokens_estimate": max_tokens,
                "estimation_method": "heuristic_chars_div_4",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "run_name": run_name,
        "model": args.model,
        "summary_variant": args.summary_variant,
        "simbench_split": args.split,
        "num_samples_per_context": args.num_samples_per_context,
        "dataset_names": dataset_names,
        "only_us_country_context": args.only_us_country_context,
        "seed": args.seed,
        "shared_twin_panel_file": str(shared_twin_panel_path),
        "request_manifest_file": str(request_manifest_path),
        "request_token_estimates_file": str(metadata_dir / "request_token_estimates.json"),
        "gold_targets_file": str(gold_targets_path),
        "selected_rows_file": str(selected_rows_path),
        "selected_contexts_file": str(selected_contexts_path),
        "batch_input_file": str(batch_path),
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(str(batch_path))


if __name__ == "__main__":
    main()
