from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR
REPO_ROOT = PACKAGE_ROOT.parent.parent
for path in (SCRIPT_DIR, PACKAGE_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import pandas as pd

from simulation_statistical.archetype_distribution_embedding.data_io.load_archetypes import load_archetype_jsonl
from simulation_statistical.archetype_distribution_embedding.data_io.load_configs import (
    load_config_table,
    load_player_game_keys,
)
from simulation_statistical.archetype_distribution_embedding.data_io.merge_sources import build_player_game_table
from simulation_statistical.archetype_distribution_embedding.evaluate.evaluate_clustering import evaluate_clustering
from simulation_statistical.archetype_distribution_embedding.evaluate.evaluate_env_model import (
    evaluate_env_predictions,
)
from simulation_statistical.archetype_distribution_embedding.features.prepare_embedding_inputs import (
    export_embedding_input_jsonl,
)
from simulation_statistical.archetype_distribution_embedding.features.reduce_embeddings import (
    join_and_reduce_embeddings,
    load_embedding_output,
)
from simulation_statistical.archetype_distribution_embedding.preprocess.normalize_tags import normalize_tag_frame
from simulation_statistical.archetype_distribution_embedding.preprocess.split_tag_blocks import (
    build_tag_block_table,
)
from simulation_statistical.archetype_distribution_embedding.preprocess.validate_records import (
    validate_clean_records,
    write_validation_report,
)
from simulation_statistical.archetype_distribution_embedding.train.fit_env_model import (
    aggregate_player_weights_to_games,
    fit_env_distribution_model,
)
from simulation_statistical.archetype_distribution_embedding.train.fit_soft_clusters import (
    fit_soft_cluster_grid,
)
from simulation_statistical.archetype_distribution_embedding.train.infer_soft_clusters import (
    infer_player_cluster_weights,
)
from simulation_statistical.archetype_distribution_embedding.utils.constants import (
    EMBEDDING_MODEL_DEFAULT,
    GMM_CLUSTER_GRID_DEFAULT,
    PCA_COMPONENTS_DEFAULT,
    REQUIRED_CONFIG_COLUMNS,
)
from simulation_statistical.archetype_distribution_embedding.utils.io_utils import (
    ensure_dir,
    load_dataframe,
    save_dataframe,
)
from simulation_statistical.archetype_distribution_embedding.utils.logging_utils import get_logger
from simulation_statistical.archetype_distribution_embedding.utils.paths import (
    DEFAULT_INPUT_PATHS,
    INTERMEDIATE_ROOT,
    MODEL_ROOT,
    OUTPUT_ROOT,
    intermediate_path,
    model_path,
    output_path,
)


LOGGER = get_logger("archetype_distribution_embedding")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the embedding-only archetype distribution pipeline."
    )
    parser.add_argument(
        "stage",
        choices=["prepare-inputs", "fit-clusters", "fit-env-model", "post-embed", "all-post-embed"],
    )

    parser.add_argument("--learn-archetypes", default=str(DEFAULT_INPUT_PATHS["learn"].archetype_jsonl))
    parser.add_argument("--learn-config", default=str(DEFAULT_INPUT_PATHS["learn"].config_csv))
    parser.add_argument("--learn-player-rounds", default=str(DEFAULT_INPUT_PATHS["learn"].player_rounds_csv))
    parser.add_argument("--val-archetypes", default=str(DEFAULT_INPUT_PATHS["val"].archetype_jsonl))
    parser.add_argument("--val-config", default=str(DEFAULT_INPUT_PATHS["val"].config_csv))
    parser.add_argument("--val-player-rounds", default=str(DEFAULT_INPUT_PATHS["val"].player_rounds_csv))

    parser.add_argument(
        "--embedding-input-learn",
        default=str(intermediate_path("embedding_input_learn.jsonl")),
    )
    parser.add_argument(
        "--embedding-input-val",
        default=str(intermediate_path("embedding_input_val.jsonl")),
    )
    parser.add_argument(
        "--embedding-output-learn",
        default=str(intermediate_path("embedding_output_learn.jsonl")),
    )
    parser.add_argument(
        "--embedding-output-val",
        default=str(intermediate_path("embedding_output_val.jsonl")),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS_DEFAULT)
    parser.add_argument(
        "--cluster-grid",
        default=",".join(str(value) for value in GMM_CLUSTER_GRID_DEFAULT),
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser


def parse_cluster_grid(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("Cluster grid must contain at least one integer")
    return values


def ensure_artifact_dirs() -> None:
    for directory in (INTERMEDIATE_ROOT, MODEL_ROOT, OUTPUT_ROOT):
        ensure_dir(directory)


def _log_merge_summary(summary: object) -> None:
    LOGGER.info(
        "%s merge: archetypes=%s player_keys=%s config_games=%s matched_player=%s unmatched_player=%s matched_config=%s unmatched_config=%s",
        summary.wave,
        summary.archetype_rows,
        summary.raw_player_game_rows,
        summary.config_game_rows,
        summary.matched_player_rows,
        summary.unmatched_player_rows,
        summary.matched_config_rows,
        summary.unmatched_config_rows,
    )


def prepare_wave_tables(
    wave: str,
    archetype_path: str,
    config_path: str,
    player_rounds_path: str,
    raw_output_path: Path,
    clean_output_path: Path,
    tag_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    archetype_df = load_archetype_jsonl(archetype_path, wave=wave)
    config_df = load_config_table(config_path, required_config_cols=list(REQUIRED_CONFIG_COLUMNS))
    player_game_keys_df = load_player_game_keys(player_rounds_path)
    merged_df, merge_summary = build_player_game_table(
        archetype_df=archetype_df,
        player_game_df=player_game_keys_df,
        config_df=config_df,
        wave=wave,
    )
    _log_merge_summary(merge_summary)
    save_dataframe(merged_df, raw_output_path)

    clean_df = normalize_tag_frame(merged_df)
    tag_blocks_df = build_tag_block_table(clean_df)
    row_diagnostics, tag_frequency_df, validation_summary = validate_clean_records(clean_df, tag_blocks_df)
    clean_df = clean_df.merge(
        row_diagnostics,
        on=["row_id", "wave", "game_id", "player_id"],
        how="left",
        validate="one_to_one",
    )

    save_dataframe(clean_df, clean_output_path)
    save_dataframe(tag_blocks_df, tag_output_path)
    return clean_df, tag_blocks_df, validation_summary, tag_frequency_df


def manual_embedding_commands(args: argparse.Namespace) -> tuple[str, str]:
    learn_command = (
        "python simulation_statistical/archetype_distribution_embedding/features/embed_openai.py "
        f"--input {args.embedding_input_learn} "
        f"--output {args.embedding_output_learn} "
        f"--model {args.embedding_model} "
        f"--batch-size {args.batch_size} "
        f"--api-key-env {args.api_key_env}"
    )
    val_command = (
        "python simulation_statistical/archetype_distribution_embedding/features/embed_openai.py "
        f"--input {args.embedding_input_val} "
        f"--output {args.embedding_output_val} "
        f"--model {args.embedding_model} "
        f"--batch-size {args.batch_size} "
        f"--api-key-env {args.api_key_env}"
    )
    return learn_command, val_command


def run_prepare_inputs(args: argparse.Namespace) -> None:
    ensure_artifact_dirs()

    learn_clean_df, learn_tag_df, learn_summary, learn_tag_freq = prepare_wave_tables(
        wave="learn",
        archetype_path=args.learn_archetypes,
        config_path=args.learn_config,
        player_rounds_path=args.learn_player_rounds,
        raw_output_path=intermediate_path("player_game_table_learn.parquet"),
        clean_output_path=intermediate_path("player_game_table_learn_clean.parquet"),
        tag_output_path=intermediate_path("tag_blocks_learn.parquet"),
    )
    val_clean_df, val_tag_df, val_summary, val_tag_freq = prepare_wave_tables(
        wave="val",
        archetype_path=args.val_archetypes,
        config_path=args.val_config,
        player_rounds_path=args.val_player_rounds,
        raw_output_path=intermediate_path("player_game_table_val.parquet"),
        clean_output_path=intermediate_path("player_game_table_val_clean.parquet"),
        tag_output_path=intermediate_path("tag_blocks_val.parquet"),
    )

    export_embedding_input_jsonl(learn_clean_df, args.embedding_input_learn)
    export_embedding_input_jsonl(val_clean_df, args.embedding_input_val)
    write_validation_report(
        intermediate_path("tag_validation_report.md"),
        learn_summary=learn_summary,
        val_summary=val_summary,
        tag_frequency_df=pd.concat([learn_tag_freq, val_tag_freq], ignore_index=True),
    )

    LOGGER.info("Saved cleaned learn rows=%s and val rows=%s", len(learn_clean_df), len(val_clean_df))
    LOGGER.info("Saved learn tag rows=%s and val tag rows=%s", len(learn_tag_df), len(val_tag_df))

    learn_command, val_command = manual_embedding_commands(args)
    LOGGER.info("Manual embedding command (learn): %s", learn_command)
    LOGGER.info("Manual embedding command (val): %s", val_command)


def run_fit_clusters(args: argparse.Namespace) -> None:
    ensure_artifact_dirs()
    learn_clean_df = load_dataframe(intermediate_path("player_game_table_learn_clean.parquet"))
    val_clean_df = load_dataframe(intermediate_path("player_game_table_val_clean.parquet"))
    learn_embedding_df = load_embedding_output(args.embedding_output_learn)
    val_embedding_df = load_embedding_output(args.embedding_output_val)

    learn_matrix_df, val_matrix_df, _ = join_and_reduce_embeddings(
        learn_player_df=learn_clean_df,
        val_player_df=val_clean_df,
        learn_embedding_df=learn_embedding_df,
        val_embedding_df=val_embedding_df,
        n_components=args.pca_components,
        standardize=True,
        pca_model_path=model_path("pca_model.pkl"),
    )
    save_dataframe(learn_matrix_df, intermediate_path("embedding_matrix_learn.parquet"))
    save_dataframe(val_matrix_df, intermediate_path("embedding_matrix_val.parquet"))

    cluster_grid = parse_cluster_grid(args.cluster_grid)
    gmm_model, diagnostics_df = fit_soft_cluster_grid(
        learn_embedding_df=learn_matrix_df,
        cluster_grid=cluster_grid,
        model_path=model_path("gmm_model.pkl"),
        diagnostics_path=output_path("gmm_diagnostics.csv"),
        random_state=args.random_state,
    )

    learn_weights_df = infer_player_cluster_weights(
        model=gmm_model,
        embedding_df=learn_matrix_df,
        output_path=output_path("player_cluster_weights_learn.parquet"),
    )
    val_weights_df = infer_player_cluster_weights(
        model=gmm_model,
        embedding_df=val_matrix_df,
        output_path=output_path("player_cluster_weights_val.parquet"),
    )

    aggregate_player_weights_to_games(
        player_weights_df=learn_weights_df,
        player_game_df=learn_clean_df,
        output_path=output_path("game_cluster_distribution_learn.parquet"),
    )
    aggregate_player_weights_to_games(
        player_weights_df=val_weights_df,
        player_game_df=val_clean_df,
        output_path=output_path("game_cluster_distribution_val.parquet"),
    )

    evaluate_clustering(
        model=gmm_model,
        diagnostics_df=diagnostics_df,
        learn_embedding_df=learn_matrix_df,
        learn_player_df=learn_clean_df,
        player_weights_df=learn_weights_df,
        summary_path=output_path("clustering_eval_summary.csv"),
        report_path=output_path("clustering_report.md"),
    )
    LOGGER.info("Finished fit-clusters with selected K=%s", gmm_model.n_components)


def run_fit_env_model(args: argparse.Namespace) -> None:
    ensure_artifact_dirs()
    learn_game_df = load_dataframe(output_path("game_cluster_distribution_learn.parquet"))
    val_game_df = load_dataframe(output_path("game_cluster_distribution_val.parquet"))

    _, learn_pred_df, val_pred_df = fit_env_distribution_model(
        learn_game_df=learn_game_df,
        val_game_df=val_game_df,
        model_path=model_path("dirichlet_env_model.pkl"),
        learn_output_path=output_path("predicted_game_cluster_distribution_learn.parquet"),
        val_output_path=output_path("predicted_game_cluster_distribution_val.parquet"),
    )
    evaluate_env_predictions(
        observed_game_df=learn_game_df,
        predicted_game_df=learn_pred_df,
        output_path=output_path("env_model_eval_summary_learn.csv"),
    )
    val_eval_df = evaluate_env_predictions(
        observed_game_df=val_game_df,
        predicted_game_df=val_pred_df,
        output_path=output_path("env_model_eval_summary_val.csv"),
    )
    combined_eval = pd.concat(
        [
            pd.read_csv(output_path("env_model_eval_summary_learn.csv")),
            val_eval_df,
        ],
        ignore_index=True,
    )
    combined_eval.to_csv(output_path("env_model_eval_summary.csv"), index=False)
    LOGGER.info("Finished fit-env-model for learn and val game distributions")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "prepare-inputs":
        run_prepare_inputs(args)
        return
    if args.stage == "fit-clusters":
        run_fit_clusters(args)
        return
    if args.stage == "fit-env-model":
        run_fit_env_model(args)
        return
    if args.stage in {"post-embed", "all-post-embed"}:
        run_fit_clusters(args)
        run_fit_env_model(args)
        return
    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
