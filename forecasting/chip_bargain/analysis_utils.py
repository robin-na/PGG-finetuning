from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common import wasserstein_distance_1d


PRIMARY_METRIC_FAMILY = "distribution_distance"

_METRIC_FAMILY_SORT_ORDER = {
    PRIMARY_METRIC_FAMILY: 0,
    "mean_alignment_abs_error": 1,
}
_DISTANCE_KIND_SORT_ORDER = {
    "wasserstein_1d": 0,
    "mean_wasserstein_1d": 1,
    "": 2,
}
_METRIC_SORT_ORDER = {
    "final_surplus_ratio": 0,
    "proposer_net_surplus": 1,
    "trade_ratio": 2,
    "acceptance_rate": 3,
    "accepted_proposer_net_surplus": 4,
    "declined_proposer_net_surplus": 5,
    "accepted_trade_ratio": 6,
    "declined_trade_ratio": 7,
    "final_total_surplus": 8,
    "mean_player_final_surplus": 9,
}

PRIMARY_HEADLINE_METRICS = [
    "final_surplus_ratio",
    "proposer_net_surplus",
    "trade_ratio",
    "acceptance_rate",
]


def _mean_alignment_rows(
    *,
    treatment_name: str,
    generated_games: pd.DataFrame,
    human_games: pd.DataFrame,
    generated_turns: pd.DataFrame,
    human_turns: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "treatment_name": treatment_name,
            "metric_scope": "game",
            "metric": "final_surplus_ratio",
            "generated_mean": float(generated_games["final_surplus_ratio"].mean()),
            "human_mean": float(human_games["final_surplus_ratio"].mean()),
        }
    )
    rows.append(
        {
            "treatment_name": treatment_name,
            "metric_scope": "turn",
            "metric": "proposer_net_surplus",
            "generated_mean": float(generated_turns["proposer_net_surplus"].mean()),
            "human_mean": float(human_turns["proposer_net_surplus"].mean()),
        }
    )
    rows.append(
        {
            "treatment_name": treatment_name,
            "metric_scope": "turn",
            "metric": "trade_ratio",
            "generated_mean": float(generated_turns["trade_ratio"].mean()),
            "human_mean": float(human_turns["trade_ratio"].mean()),
        }
    )
    rows.append(
        {
            "treatment_name": treatment_name,
            "metric_scope": "turn",
            "metric": "acceptance_rate",
            "generated_mean": float(generated_turns["accepted_binary"].mean()),
            "human_mean": float(human_turns["accepted_binary"].mean()),
        }
    )
    for row in rows:
        row["abs_error"] = abs(row["generated_mean"] - row["human_mean"])
    return rows


def _distribution_rows(
    *,
    treatment_name: str,
    generated_games: pd.DataFrame,
    human_games: pd.DataFrame,
    generated_turns: pd.DataFrame,
    human_turns: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append_row(metric: str, x: pd.Series, y: pd.Series) -> None:
        rows.append(
            {
                "treatment_name": treatment_name,
                "metric_family": PRIMARY_METRIC_FAMILY,
                "metric": metric,
                "distance_kind": "wasserstein_1d",
                "score": wasserstein_distance_1d(x, y),
            }
        )

    append_row("final_surplus_ratio", generated_games["final_surplus_ratio"], human_games["final_surplus_ratio"])
    append_row("final_total_surplus", generated_games["final_total_surplus"], human_games["final_total_surplus"])
    append_row(
        "mean_player_final_surplus",
        generated_games["mean_player_final_surplus"],
        human_games["mean_player_final_surplus"],
    )
    append_row("proposer_net_surplus", generated_turns["proposer_net_surplus"], human_turns["proposer_net_surplus"])
    append_row("trade_ratio", generated_turns["trade_ratio"], human_turns["trade_ratio"])
    append_row("acceptance_rate", generated_turns["accepted_binary"], human_turns["accepted_binary"])

    generated_accepted = generated_turns[generated_turns["accepted_binary"] == 1]
    human_accepted = human_turns[human_turns["accepted_binary"] == 1]
    if not generated_accepted.empty and not human_accepted.empty:
        append_row(
            "accepted_proposer_net_surplus",
            generated_accepted["proposer_net_surplus"],
            human_accepted["proposer_net_surplus"],
        )
        append_row(
            "accepted_trade_ratio",
            generated_accepted["trade_ratio"],
            human_accepted["trade_ratio"],
        )

    generated_declined = generated_turns[generated_turns["accepted_binary"] == 0]
    human_declined = human_turns[human_turns["accepted_binary"] == 0]
    if not generated_declined.empty and not human_declined.empty:
        append_row(
            "declined_proposer_net_surplus",
            generated_declined["proposer_net_surplus"],
            human_declined["proposer_net_surplus"],
        )
        append_row(
            "declined_trade_ratio",
            generated_declined["trade_ratio"],
            human_declined["trade_ratio"],
        )

    return rows


def compute_metric_tables(
    *,
    generated_game_records: pd.DataFrame,
    human_game_records: pd.DataFrame,
    generated_turn_records: pd.DataFrame,
    human_turn_records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_treatments = sorted(
        set(generated_game_records["treatment_name"].dropna().astype(str))
        & set(human_game_records["treatment_name"].dropna().astype(str))
    )

    mean_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    for treatment_name in common_treatments:
        generated_games = generated_game_records[generated_game_records["treatment_name"] == treatment_name]
        human_games = human_game_records[human_game_records["treatment_name"] == treatment_name]
        generated_turns = generated_turn_records[generated_turn_records["treatment_name"] == treatment_name]
        human_turns = human_turn_records[human_turn_records["treatment_name"] == treatment_name]
        if generated_games.empty or human_games.empty or generated_turns.empty or human_turns.empty:
            continue
        mean_rows.extend(
            _mean_alignment_rows(
                treatment_name=treatment_name,
                generated_games=generated_games,
                human_games=human_games,
                generated_turns=generated_turns,
                human_turns=human_turns,
            )
        )
        dist_rows.extend(
            _distribution_rows(
                treatment_name=treatment_name,
                generated_games=generated_games,
                human_games=human_games,
                generated_turns=generated_turns,
                human_turns=human_turns,
            )
        )

    mean_df = pd.DataFrame(mean_rows)
    dist_df = pd.DataFrame(dist_rows)

    overall_rows: list[dict[str, Any]] = []
    if not dist_df.empty:
        for (metric_family, metric, distance_kind), group in dist_df.groupby(
            ["metric_family", "metric", "distance_kind"],
            sort=False,
        ):
            overall_rows.append(
                {
                    "metric_family": str(metric_family),
                    "metric": str(metric),
                    "distance_kind": str(distance_kind),
                    "n_treatments": int(group["treatment_name"].nunique()),
                    "mean_value": float(group["score"].mean()),
                    "median_value": float(group["score"].median()),
                }
            )
    overall_df = pd.DataFrame(overall_rows)
    return mean_df, dist_df, overall_df


def sort_overall_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["_metric_family_order"] = frame["metric_family"].map(_METRIC_FAMILY_SORT_ORDER).fillna(999)
    frame["_metric_order"] = frame["metric"].map(_METRIC_SORT_ORDER).fillna(999)
    frame["_distance_kind_order"] = frame["distance_kind"].map(_DISTANCE_KIND_SORT_ORDER).fillna(999)
    frame = frame.sort_values(
        ["_metric_family_order", "_metric_order", "_distance_kind_order", "metric", "distance_kind"],
        kind="stable",
    ).drop(columns=["_metric_family_order", "_metric_order", "_distance_kind_order"])
    return frame.reset_index(drop=True)


def build_primary_distribution_summary(overall_df: pd.DataFrame) -> pd.DataFrame:
    if overall_df.empty:
        return overall_df
    primary = overall_df[
        (overall_df["metric_family"] == PRIMARY_METRIC_FAMILY)
        & (overall_df["metric"].isin(PRIMARY_HEADLINE_METRICS))
    ].copy()
    return sort_overall_metric_summary(primary)
