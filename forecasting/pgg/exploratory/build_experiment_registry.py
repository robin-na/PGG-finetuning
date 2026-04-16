from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


PGG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_DIR = PGG_ROOT / "registry"
METADATA_DIR = PGG_ROOT / "metadata"
RESULTS_DIR = PGG_ROOT / "results"

CORE_RUN_NAMES = {
    "baseline_gpt_5_1",
    "baseline_gpt_5_mini",
    "demographic_only_row_resampled_seed_0_gpt_5_1",
    "demographic_only_row_resampled_seed_0_gpt_5_mini",
    "twin_sampled_seed_0_gpt_5_1",
    "twin_sampled_seed_0_gpt_5_mini",
    "twin_sampled_unadjusted_seed_0_gpt_5_1",
    "twin_sampled_unadjusted_seed_0_gpt_5_mini",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rooted(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        return str(path)
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return str(path)
    return str(Path("PGG-finetuning") / rel)


def _canonical_variant(raw_variant: str | None) -> str:
    value = (raw_variant or "").strip()
    mapping = {
        "baseline_direct_transcript": "baseline",
        "baseline": "baseline",
        "twin_sampled_seed_0": "twin_sampled_seed_0",
        "twin-sampled_seed_0": "twin_sampled_seed_0",
        "twin_sampled_unadjusted_seed_0": "twin_sampled_unadjusted_seed_0",
        "twin-sampled_unadjusted_seed_0": "twin_sampled_unadjusted_seed_0",
        "demographic_only_row_resampled_seed_0": "demographic_only_row_resampled_seed_0",
    }
    return mapping.get(value, value)


def _variant_family(variant: str) -> tuple[str, str, str]:
    if variant == "baseline":
        return ("baseline", "none", "No player-profile augmentation.")
    if variant == "demographic_only_row_resampled_seed_0":
        return (
            "demographic_only",
            "pgg_validation_demographics",
            "Synthetic demographic-only cards resampled from validation-wave PGG demographic rows.",
        )
    if variant == "twin_sampled_seed_0":
        return (
            "twin_corrected",
            "twin_profiles",
            "Twin-derived persona cards with aggregate age/sex/education correction to the validation PGG distribution.",
        )
    if variant == "twin_sampled_unadjusted_seed_0":
        return (
            "twin_unadjusted",
            "twin_profiles",
            "Twin-derived persona cards without demographic correction.",
        )
    return ("other", "unknown", "Non-core or legacy run.")


def _infer_model_label(run_name: str, manifest: dict[str, Any]) -> str:
    model = str(manifest.get("model") or "").strip()
    if model:
        return model
    lowered = run_name.lower()
    if "gpt_5_mini" in lowered:
        return "gpt-5-mini"
    if "gpt_5_1" in lowered:
        return "gpt-5.1"
    return "unknown"


def _result_manifest(run_name: str, suffix: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{run_name}{suffix}" / "manifest.json"
    if not path.exists():
        return None
    return _read_json(path)


def _experiment_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(METADATA_DIR.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        run_name = str(manifest["run_name"])
        raw_variant = str(manifest.get("variant_name_raw") or manifest.get("variant_name") or "")
        canonical_variant = _canonical_variant(raw_variant)
        variant_family, profile_source, variant_description = _variant_family(canonical_variant)
        model = _infer_model_label(run_name, manifest)

        vs_human_manifest = _result_manifest(run_name, "__vs_human_treatments")
        gold_manifest = _result_manifest(run_name, "__gold_eval")

        total_requests = int(manifest.get("total_request_count") or manifest.get("total_requests") or 0)
        generated_games = (
            int(vs_human_manifest["num_generated_games"])
            if vs_human_manifest and "num_generated_games" in vs_human_manifest
            else None
        )
        generated_gap = (total_requests - generated_games) if generated_games is not None else None

        notes: list[str] = []
        if run_name not in CORE_RUN_NAMES:
            notes.append(
                "Legacy or duplicate metadata artifact; not part of the canonical 8-run comparison set."
            )
        if raw_variant and canonical_variant and raw_variant != canonical_variant:
            notes.append("Registry normalizes a non-canonical manifest variant name.")
        if canonical_variant == "baseline":
            notes.append(
                "Baseline uses one exemplar game per treatment and repeats it to match valid-start treatment counts."
            )
        elif canonical_variant in {
            "demographic_only_row_resampled_seed_0",
            "twin_sampled_seed_0",
            "twin_sampled_unadjusted_seed_0",
        }:
            notes.append("Augmented runs use the full 417 assigned validation games from seat-level assignment manifests.")
        if generated_gap is not None and generated_gap > 0:
            notes.append(
                f"Generated/evaluable game count is {generated_gap} lower than total requested games."
            )
        if vs_human_manifest is not None:
            notes.append(
                "Human reference set uses validation games with valid_number_of_starting_players == True and can still include incomplete games."
            )

        rows.append(
            {
                "run_name": run_name,
                "is_core_run": run_name in CORE_RUN_NAMES,
                "model": model,
                "raw_variant_name": raw_variant or None,
                "canonical_variant": canonical_variant,
                "variant_family": variant_family,
                "profile_source": profile_source,
                "variant_description": variant_description,
                "split": manifest.get("split"),
                "selection_mode": manifest.get("selection_mode"),
                "selection_rule": manifest.get("selection_rule"),
                "repeat_count_mode": manifest.get("repeat_count_mode"),
                "selected_game_count": manifest.get("selected_game_count"),
                "total_request_count": total_requests,
                "require_valid_starting_players": manifest.get("require_valid_starting_players"),
                "persona_assignment_file": _repo_rooted(manifest.get("persona_assignment_file")),
                "persona_cards_file": _repo_rooted(manifest.get("persona_cards_file")),
                "persona_shared_notes_file": _repo_rooted(manifest.get("persona_shared_notes_file")),
                "batch_input_file": _repo_rooted(manifest.get("batch_input_file")),
                "expected_batch_output_file": _repo_rooted(manifest.get("expected_batch_output_file")),
                "batch_output_exists": Path(str(manifest.get("expected_batch_output_file", ""))).exists()
                if manifest.get("expected_batch_output_file")
                else False,
                "parsed_output_exists": (manifest_path.parent / "parsed_output.jsonl").exists(),
                "vs_human_results_dir": _repo_rooted(
                    str((RESULTS_DIR / f"{run_name}__vs_human_treatments"))
                )
                if (RESULTS_DIR / f"{run_name}__vs_human_treatments").exists()
                else None,
                "vs_human_result_manifest_exists": vs_human_manifest is not None,
                "vs_human_num_generated_games": generated_games,
                "vs_human_num_human_games": (
                    int(vs_human_manifest["num_human_games"])
                    if vs_human_manifest and "num_human_games" in vs_human_manifest
                    else None
                ),
                "vs_human_generated_game_gap": generated_gap,
                "gold_eval_results_dir": _repo_rooted(str((RESULTS_DIR / f"{run_name}__gold_eval")))
                if (RESULTS_DIR / f"{run_name}__gold_eval").exists()
                else None,
                "gold_eval_manifest_exists": gold_manifest is not None,
                "gold_eval_evaluated_requests": (
                    int(gold_manifest["evaluated_requests"])
                    if gold_manifest and "evaluated_requests" in gold_manifest
                    else None
                ),
                "notes": " | ".join(notes) if notes else None,
            }
        )
    return rows


def _analysis_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    model_comparison_manifest = _result_manifest("model_comparison", "__noise_ceiling")
    # The combined comparison uses a fixed output name, not run-name suffixing.
    model_comparison_path = RESULTS_DIR / "model_comparison__noise_ceiling" / "manifest.json"
    if model_comparison_path.exists():
        model_comparison_manifest = _read_json(model_comparison_path)

    micro_manifest_path = RESULTS_DIR / "micro_distribution_alignment__llms" / "manifest.json"
    micro_manifest = _read_json(micro_manifest_path) if micro_manifest_path.exists() else None
    macro_manifest_path = RESULTS_DIR / "macro_pointwise_alignment__llms" / "manifest.json"
    macro_manifest = _read_json(macro_manifest_path) if macro_manifest_path.exists() else None

    rows.append(
        {
            "analysis_id": "vs_human_treatments_per_run",
            "script": "forecasting/pgg/analyze_vs_human_treatments.py",
            "output_location": "forecasting/pgg/results/<run_name>__vs_human_treatments/",
            "analysis_level": "macro / per-treatment game-distribution comparison",
            "unit_of_comparison": "generated games vs human validation games within CONFIG_treatmentName",
            "metrics_or_outputs": (
                "game, actor, and round summaries; treatment means; treatment dispersion; treatment Wasserstein distance; typicality"
            ),
            "noise_ceiling_method": "none in this step",
            "current_scope": "All available core runs have per-run treatment-comparison outputs.",
            "risk_or_note": (
                "Human reference selection is valid-start validation games, but still allows incomplete games; that should be treated as an explicit design choice."
            ),
        }
    )

    rows.append(
        {
            "analysis_id": "model_comparison_noise_ceiling",
            "script": "forecasting/pgg/compare_models_with_noise_ceiling.py",
            "output_location": "forecasting/pgg/results/model_comparison__noise_ceiling/",
            "analysis_level": "macro / model-level summary against human noise ceiling",
            "unit_of_comparison": "treatment-level scalar metric summaries",
            "metrics_or_outputs": (
                "RMSE of treatment means, mean Wasserstein distance, mean |log SD ratio|, mean |log IQR ratio|"
            ),
            "noise_ceiling_method": (
                "For each CONFIG_treatmentName, bootstrap two independent human resamples with replacement; one pseudo-model sample uses the shared generated count and one pseudo-human sample uses the human count."
            ),
            "current_scope": (
                f"Current committed output compares runs {model_comparison_manifest.get('run_names') if model_comparison_manifest else '[]'}."
            ),
            "risk_or_note": (
                "Scores are computed after trimming each model to the shared generated-game count within each treatment."
            ),
        }
    )

    rows.append(
        {
            "analysis_id": "macro_pointwise_alignment",
            "script": "forecasting/pgg/exploratory/plot_macro_pointwise_alignment.py",
            "output_location": "forecasting/pgg/results/macro_pointwise_alignment__llms/",
            "analysis_level": "macro / pointwise across the 40 validation configs",
            "unit_of_comparison": "config-level means and within-config distribution distances",
            "metrics_or_outputs": (
                "RMSE of config means and mean within-config Wasserstein distance for mean, first-round, and final-round contribution and efficiency"
            ),
            "noise_ceiling_method": (
                "Within each CONFIG_treatmentName, bootstrap two independent human resamples with replacement and compare their config means and within-config Wasserstein distances."
            ),
            "current_scope": (
                f"Current committed manifest covers runs {macro_manifest.get('run_names') if macro_manifest else 'missing'}."
            ),
            "risk_or_note": None if macro_manifest is not None else "Macro output directory exists, but no manifest was found.",
        }
    )

    rows.append(
        {
            "analysis_id": "micro_distribution_alignment",
            "script": "forecasting/pgg/exploratory/analyze_micro_distribution_alignment.py",
            "output_location": "forecasting/pgg/results/micro_distribution_alignment__llms/",
            "analysis_level": "micro / within-config player and round distributions",
            "unit_of_comparison": "within-treatment Wasserstein distances over player-level and round-level summaries",
            "metrics_or_outputs": (
                "player mean contribution, player mean payoff, round contribution, round efficiency, and optional round-to-round delta distributions"
            ),
            "noise_ceiling_method": (
                "Optional bootstrap: resample whole human games within each CONFIG_treatmentName, derive pseudo-generated and pseudo-human player/round summaries, then compute within-config Wasserstein distances."
            ),
            "current_scope": (
                f"Current committed manifest has bootstrap_iters={micro_manifest.get('bootstrap_iters') if micro_manifest else 'missing'}."
            ),
            "risk_or_note": (
                None
                if micro_manifest and int(micro_manifest.get("bootstrap_iters", 0)) > 0
                else "Current committed micro output has no materialized noise ceiling because bootstrap_iters is 0."
            ),
        }
    )

    rows.append(
        {
            "analysis_id": "twin_profile_behavior_links",
            "script": "forecasting/pgg/exploratory/analyze_twin_profile_behavior_links.py",
            "output_location": "forecasting/pgg/results/twin_profile_behavior_links/",
            "analysis_level": "exploratory / profile-to-behavior linkage",
            "unit_of_comparison": "Twin cue scores vs simulated player-level outcomes",
            "metrics_or_outputs": (
                "top cue-outcome associations for contribution, payoff, volatility, punishment, reward, and messaging"
            ),
            "noise_ceiling_method": "none",
            "current_scope": "Current committed output covers Twin-corrected runs only.",
            "risk_or_note": "This is exploratory analysis, not a primary forecasting benchmark.",
        }
    )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    experiment_rows = _experiment_rows()
    analysis_rows = _analysis_rows()

    _write_csv(REGISTRY_DIR / "experiment_registry.csv", experiment_rows)
    (REGISTRY_DIR / "experiment_registry.json").write_text(
        json.dumps(experiment_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(REGISTRY_DIR / "analysis_registry.csv", analysis_rows)
    (REGISTRY_DIR / "analysis_registry.json").write_text(
        json.dumps(analysis_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote registry files to {REGISTRY_DIR}")


if __name__ == "__main__":
    main()
