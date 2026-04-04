from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import transforms


RUN_SPECS = [
    ("twin_sampled_seed_0_gpt_5_1", "gpt-5.1 twin"),
    ("twin_sampled_seed_0_gpt_5_mini", "gpt-5-mini twin"),
]

CUE_LABELS = {
    "behavioral_stability": "Behavioral stability",
    "communication_coordination": "Communication coordination",
    "conditional_cooperation": "Conditional cooperation",
    "cooperation_orientation": "Cooperation orientation",
    "exploitation_caution": "Exploitation caution",
    "generosity_without_return": "Generosity without return",
    "norm_enforcement": "Norm enforcement",
}

OUTCOME_SPECS = [
    ("player_mean_contribution_rate", "Mean contribution", None),
    ("player_mean_normalized_payoff", "Mean normalized payoff", None),
    ("player_contribution_volatility", "Contribution volatility", None),
    ("player_punish_round_rate", "Punish rate", "punishment_enabled"),
    ("player_reward_round_rate", "Reward rate", "reward_enabled"),
    ("player_message_rate", "Message rate", "chat_enabled"),
]

MODEL_COLORS = {
    "gpt-5.1 twin": "#2b6ea6",
    "gpt-5-mini twin": "#e37a1f",
}

CUE_TEXT_COLOR = "#255785"
OUTCOME_TEXT_COLOR = "#b45f14"
ARROW_TEXT_COLOR = "#6b6b6b"


def _load_assignment_df(assignment_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with assignment_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            game_id = str(record["gameId"])
            treatment_name = str(record["CONFIG_treatmentName"])
            for assignment in record["assignments"]:
                twin_pid = assignment.get("twin_pid")
                player_id = assignment.get("pgg_roster_playerId")
                seat_index = assignment.get("seat_index")
                if twin_pid is None or player_id is None or seat_index is None:
                    continue
                rows.append(
                    {
                        "game_id": game_id,
                        "treatment_name": treatment_name,
                        "seat_index": int(seat_index),
                        "player_id": str(player_id),
                        "twin_pid": str(twin_pid),
                    }
                )
    return pd.DataFrame(rows)


def _load_cue_df(cards_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with cards_path.open(encoding="utf-8") as handle:
        for line in handle:
            card = json.loads(line)
            participant = card.get("participant", {})
            twin_pid = participant.get("pid")
            if twin_pid is None:
                continue
            row: dict[str, Any] = {"twin_pid": str(twin_pid)}
            for cue in card.get("transfer_relevance", []):
                cue_name = cue.get("cue")
                score = cue.get("score_0_to_100")
                if cue_name is None or score is None:
                    continue
                row[cue_name] = float(score) / 100.0
            rows.append(row)
    cue_df = pd.DataFrame(rows)
    expected_columns = ["twin_pid", *sorted(CUE_LABELS)]
    for cue_name in CUE_LABELS:
        if cue_name not in cue_df.columns:
            cue_df[cue_name] = np.nan
    return cue_df[expected_columns].drop_duplicates("twin_pid")


def _build_avatar_player_map(assignment_df: pd.DataFrame, request_manifest_df: pd.DataFrame) -> pd.DataFrame:
    request_subset = request_manifest_df[["custom_id", "game_id", "avatars"]].copy()
    request_subset["avatars"] = request_subset["avatars"].apply(json.loads)
    avatar_rows: list[dict[str, Any]] = []
    for row in request_subset.to_dict(orient="records"):
        avatars = row["avatars"]
        game_assignments = assignment_df[assignment_df["game_id"] == row["game_id"]].sort_values("seat_index")
        if len(avatars) != len(game_assignments):
            continue
        for avatar, assignment in zip(avatars, game_assignments.to_dict(orient="records"), strict=True):
            avatar_rows.append(
                {
                    "custom_id": str(row["custom_id"]),
                    "game_id": str(row["game_id"]),
                    "avatar": str(avatar),
                    "player_id": str(assignment["player_id"]),
                    "twin_pid": str(assignment["twin_pid"]),
                    "seat_index": int(assignment["seat_index"]),
                    "treatment_name": str(assignment["treatment_name"]),
                }
            )
    return pd.DataFrame(avatar_rows)


def _build_message_df(parsed_output_path: Path, avatar_map_df: pd.DataFrame) -> pd.DataFrame:
    avatar_lookup = {
        (str(row["custom_id"]), str(row["avatar"])): str(row["player_id"])
        for row in avatar_map_df.to_dict(orient="records")
    }
    message_rows: list[dict[str, Any]] = []
    with parsed_output_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            custom_id = str(record["custom_id"])
            message_count_by_player: dict[str, int] = {}
            for round_record in record.get("predicted_rounds", []):
                for message in round_record.get("messages", []):
                    speaker = str(message.get("speaker", ""))
                    player_id = avatar_lookup.get((custom_id, speaker))
                    if player_id is None:
                        continue
                    message_count_by_player[player_id] = message_count_by_player.get(player_id, 0) + 1
            for player_id, message_count in message_count_by_player.items():
                message_rows.append(
                    {
                        "custom_id": custom_id,
                        "player_id": player_id,
                        "message_count": int(message_count),
                    }
                )
    if not message_rows:
        return pd.DataFrame(columns=["custom_id", "player_id", "message_count"])
    return pd.DataFrame(message_rows)


def _build_actor_outcome_df(result_dir: Path, metadata_dir: Path, assignment_df: pd.DataFrame, cue_df: pd.DataFrame) -> pd.DataFrame:
    generated_actor_df = pd.read_csv(result_dir / "generated_actor_summary.csv")
    generated_game_df = pd.read_csv(result_dir / "generated_game_summary.csv")
    request_manifest_df = pd.read_csv(metadata_dir / "request_manifest.csv")
    avatar_map_df = _build_avatar_player_map(assignment_df, request_manifest_df)
    message_df = _build_message_df(metadata_dir / "parsed_output.jsonl", avatar_map_df)

    actor_summary_df = (
        generated_actor_df.groupby(["custom_id", "player_id"], as_index=False)
        .agg(
            player_mean_contribution_rate=("contribution_rate", "mean"),
            player_mean_normalized_payoff=("round_normalized_payoff", "mean"),
            player_contribution_volatility=("contribution_rate", lambda values: float(pd.Series(values).std(ddof=0))),
            player_punish_round_rate=("has_punish", "mean"),
            player_reward_round_rate=("has_reward", "mean"),
            observed_rounds=("round_number", "nunique"),
        )
    )
    game_features_df = generated_game_df[
        [
            "custom_id",
            "game_id",
            "treatment_name",
            "chat_enabled",
            "punishment_enabled",
            "reward_enabled",
            "num_rounds",
        ]
    ].copy()
    joined = actor_summary_df.merge(game_features_df, on="custom_id", how="left")
    joined = joined.merge(assignment_df[["game_id", "player_id", "twin_pid", "seat_index"]], on=["game_id", "player_id"], how="left")
    joined = joined.merge(
        message_df,
        on=["custom_id", "player_id"],
        how="left",
    )
    joined["message_count"] = joined["message_count"].fillna(0).astype(int)
    joined["player_message_rate"] = joined["message_count"] / joined["observed_rounds"].replace(0, np.nan)
    joined = joined.merge(cue_df, on="twin_pid", how="left")
    return joined


def _within_game_effect(sub_df: pd.DataFrame, cue_name: str, outcome_name: str) -> dict[str, Any] | None:
    current = sub_df[["custom_id", "twin_pid", cue_name, outcome_name]].dropna().copy()
    if current.empty:
        return None
    current["x_dm"] = current[cue_name] - current.groupby("custom_id")[cue_name].transform("mean")
    current["y_dm"] = current[outcome_name] - current.groupby("custom_id")[outcome_name].transform("mean")
    if float(current["x_dm"].abs().sum()) == 0.0 or float(current["y_dm"].abs().sum()) == 0.0:
        return None

    x_sd = float(current["x_dm"].std(ddof=1))
    y_sd = float(current["y_dm"].std(ddof=1))
    if not np.isfinite(x_sd) or not np.isfinite(y_sd) or x_sd == 0.0 or y_sd == 0.0:
        return None

    current["x_std"] = current["x_dm"] / x_sd
    current["y_std"] = current["y_dm"] / y_sd
    model = sm.OLS(current["y_std"], current[["x_std"]])
    result = model.fit(cov_type="cluster", cov_kwds={"groups": current["custom_id"]})

    beta = float(result.params["x_std"])
    stderr = float(result.bse["x_std"])
    pvalue = float(result.pvalues["x_std"])
    return {
        "beta": beta,
        "stderr": stderr,
        "pvalue": pvalue,
        "ci_low": beta - 1.96 * stderr,
        "ci_high": beta + 1.96 * stderr,
        "num_seats": int(current.shape[0]),
        "num_games": int(current["custom_id"].nunique()),
        "num_twin_profiles": int(current["twin_pid"].nunique()),
    }


def _compute_effects(seat_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome_name, outcome_label, eligibility_col in OUTCOME_SPECS:
        current = seat_df.copy()
        if eligibility_col is not None:
            current = current[current[eligibility_col].astype(bool)].copy()
        for cue_name, cue_label in CUE_LABELS.items():
            effect = _within_game_effect(current, cue_name, outcome_name)
            if effect is None:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "cue": cue_name,
                    "cue_label": cue_label,
                    "outcome": outcome_name,
                    "outcome_label": outcome_label,
                    "eligibility_filter": eligibility_col or "all_games",
                    **effect,
                }
            )
    return pd.DataFrame(rows)


def _select_top_effects(effect_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    pivot = (
        effect_df.pivot_table(
            index=["cue", "cue_label", "outcome", "outcome_label"],
            columns="model_name",
            values="beta",
            aggfunc="first",
        )
        .reset_index()
    )
    required_models = [label for _, label in RUN_SPECS]
    pivot = pivot.dropna(subset=required_models).copy()
    if pivot.empty:
        return pivot
    sign_product = np.sign(pivot[required_models[0]]) * np.sign(pivot[required_models[1]])
    pivot = pivot[sign_product > 0].copy()
    if pivot.empty:
        return pivot
    pivot["min_abs_beta"] = pivot[required_models].abs().min(axis=1)
    pivot["mean_abs_beta"] = pivot[required_models].abs().mean(axis=1)
    ranked = pivot.sort_values(["min_abs_beta", "mean_abs_beta"], ascending=False).copy()

    priority_outcomes = [
        "player_mean_contribution_rate",
        "player_mean_normalized_payoff",
        "player_punish_round_rate",
        "player_reward_round_rate",
        "player_message_rate",
    ]
    selected_frames: list[pd.DataFrame] = []
    selected_pairs: set[tuple[str, str]] = set()
    for outcome_name in priority_outcomes:
        outcome_rows = ranked[ranked["outcome"] == outcome_name]
        if outcome_rows.empty:
            continue
        top_row = outcome_rows.head(1).copy()
        pair_key = (str(top_row.iloc[0]["cue"]), str(top_row.iloc[0]["outcome"]))
        if pair_key in selected_pairs:
            continue
        selected_pairs.add(pair_key)
        selected_frames.append(top_row)

    for row in ranked.to_dict(orient="records"):
        if len(selected_pairs) >= top_k:
            break
        pair_key = (str(row["cue"]), str(row["outcome"]))
        if pair_key in selected_pairs:
            continue
        selected_pairs.add(pair_key)
        selected_frames.append(pd.DataFrame([row]))

    top = pd.concat(selected_frames, ignore_index=True) if selected_frames else ranked.head(top_k).copy()
    top = top.sort_values(["min_abs_beta", "mean_abs_beta"], ascending=False).head(top_k).copy()
    return top


def _best_pairs_by_outcome(effect_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        effect_df.pivot_table(
            index=["cue", "cue_label", "outcome", "outcome_label"],
            columns="model_name",
            values="beta",
            aggfunc="first",
        )
        .reset_index()
    )
    required_models = [label for _, label in RUN_SPECS]
    pivot = pivot.dropna(subset=required_models).copy()
    pivot = pivot[(pivot[required_models[0]] * pivot[required_models[1]]) > 0].copy()
    if pivot.empty:
        return pivot
    pivot["min_abs_beta"] = pivot[required_models].abs().min(axis=1)
    pivot["mean_abs_beta"] = pivot[required_models].abs().mean(axis=1)
    return (
        pivot.sort_values(["outcome", "min_abs_beta", "mean_abs_beta"], ascending=[True, False, False])
        .groupby("outcome", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def _plot_top_effects(effect_df: pd.DataFrame, top_pairs_df: pd.DataFrame, output_path: Path) -> None:
    if top_pairs_df.empty:
        return
    current = effect_df.merge(
        top_pairs_df[["cue", "outcome"]],
        on=["cue", "outcome"],
        how="inner",
    ).copy()
    ordering = top_pairs_df[["cue", "outcome", "cue_label", "outcome_label"]].copy()
    ordering["pair_label"] = ordering["cue_label"] + " \u27f6 " + ordering["outcome_label"]
    pair_order = ordering["pair_label"].tolist()
    current["pair_label"] = current["cue_label"] + " \u27f6 " + current["outcome_label"]
    current["pair_label"] = pd.Categorical(current["pair_label"], categories=pair_order, ordered=True)
    current = current.sort_values(["pair_label", "model_name"]).copy()

    fig = plt.figure(figsize=(13.2, max(4.8, 0.64 * len(pair_order) + 1.15)), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.65, 4.35], wspace=0.015)
    label_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1], sharey=label_ax)
    y = np.arange(len(pair_order), dtype=float)
    offsets = {
        "gpt-5.1 twin": -0.13,
        "gpt-5-mini twin": 0.13,
    }
    for model_name, color in MODEL_COLORS.items():
        model_df = current[current["model_name"] == model_name].set_index("pair_label").reindex(pair_order)
        positions = y + offsets[model_name]
        betas = model_df["beta"].to_numpy(dtype=float)
        lower = betas - model_df["ci_low"].to_numpy(dtype=float)
        upper = model_df["ci_high"].to_numpy(dtype=float) - betas
        ax.errorbar(
            betas,
            positions,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=3.5,
            markersize=6.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=model_name,
        )

    ci_low = current["ci_low"].min()
    ci_high = current["ci_high"].max()
    x_pad = max(0.05, 0.06 * float(ci_high - ci_low))
    ax.set_xlim(float(ci_low - x_pad), float(ci_high + x_pad))
    ax.axvline(0.0, color="#6d6d6d", linewidth=1.2, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(pair_order))
    ax.invert_yaxis()
    ax.set_xlabel("Within-game standardized effect (95% CI)")
    ax.grid(axis="x", color="#d9dde2", linewidth=0.9, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="x", labelsize=10.5, colors="#222222")
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, -0.12),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.6,
        columnspacing=1.4,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.17, top=0.865, wspace=0.015)

    label_ax.set_xlim(0.0, 1.0)
    label_ax.set_ylim(ax.get_ylim())
    label_ax.axis("off")
    text_transform = transforms.blended_transform_factory(label_ax.transAxes, label_ax.transData)
    for row_index, (_, row) in enumerate(ordering.iterrows()):
        y_pos = y[row_index]
        label_ax.text(
            0.46,
            y_pos,
            str(row["cue_label"]),
            transform=text_transform,
            ha="right",
            va="center",
            fontsize=10.4,
            color=CUE_TEXT_COLOR,
            clip_on=False,
        )
        label_ax.text(
            0.53,
            y_pos,
            "\u27f6",
            transform=text_transform,
            ha="center",
            va="center",
            fontsize=11.2,
            color=ARROW_TEXT_COLOR,
            clip_on=False,
        )
        label_ax.text(
            0.59,
            y_pos,
            str(row["outcome_label"]),
            transform=text_transform,
            ha="left",
            va="center",
            fontsize=10.4,
            color=OUTCOME_TEXT_COLOR,
            clip_on=False,
        )
    ax_pos = ax.get_position()
    title_left = ax_pos.x0
    title_width = ax_pos.x1 - ax_pos.x0
    title_y = ax_pos.y1 + 0.028
    fig.text(
        title_left + 0.40 * title_width,
        title_y,
        "Behavioral profile",
        ha="right",
        va="center",
        fontsize=15,
        color=CUE_TEXT_COLOR,
        fontweight="bold",
    )
    fig.text(
        title_left + 0.50 * title_width,
        title_y,
        "\u2192",
        ha="center",
        va="center",
        fontsize=15,
        color=ARROW_TEXT_COLOR,
    )
    fig.text(
        title_left + 0.60 * title_width,
        title_y,
        "Simulated PGG behavior",
        ha="left",
        va="center",
        fontsize=15,
        color=OUTCOME_TEXT_COLOR,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze relationships between Twin profile cues and simulated PGG behavior.")
    parser.add_argument("--forecasting-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.forecasting_root / "results" / "twin_profile_behavior_links"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assignment_df = _load_assignment_df(
        args.repo_root
        / "non-PGG_generalization"
        / "task_grounding"
        / "output"
        / "twin_to_pgg_validation_persona_sampling"
        / "seed_0"
        / "game_assignments.jsonl"
    )
    cue_df = _load_cue_df(
        args.repo_root
        / "non-PGG_generalization"
        / "task_grounding"
        / "output"
        / "twin_extended_profile_cards"
        / "pgg_prompt_min"
        / "twin_extended_profile_cards.jsonl"
    )

    seat_frames: list[pd.DataFrame] = []
    effect_frames: list[pd.DataFrame] = []
    for run_name, model_name in RUN_SPECS:
        result_dir = args.forecasting_root / "results" / f"{run_name}__vs_human_treatments"
        metadata_dir = args.forecasting_root / "metadata" / run_name
        seat_df = _build_actor_outcome_df(result_dir, metadata_dir, assignment_df, cue_df)
        seat_df["model_name"] = model_name
        seat_frames.append(seat_df)
        effect_frames.append(_compute_effects(seat_df, model_name))

    seat_df = pd.concat(seat_frames, ignore_index=True)
    effect_df = pd.concat(effect_frames, ignore_index=True)
    top_pairs_df = _select_top_effects(effect_df, args.top_k)
    best_by_outcome_df = _best_pairs_by_outcome(effect_df)
    top_effect_df = effect_df.merge(top_pairs_df[["cue", "outcome"]], on=["cue", "outcome"], how="inner")

    seat_df.sort_values(["model_name", "custom_id", "player_id"]).to_csv(
        args.output_dir / "seat_level_profile_behavior.csv",
        index=False,
    )
    effect_df.sort_values(["cue", "outcome", "model_name"]).to_csv(
        args.output_dir / "profile_behavior_effects_full.csv",
        index=False,
    )
    top_pairs_df.sort_values(["min_abs_beta", "mean_abs_beta"], ascending=False).to_csv(
        args.output_dir / "profile_behavior_top_pairs.csv",
        index=False,
    )
    best_by_outcome_df.sort_values("outcome").to_csv(
        args.output_dir / "profile_behavior_best_by_outcome.csv",
        index=False,
    )
    top_effect_df.sort_values(["cue", "outcome", "model_name"]).to_csv(
        args.output_dir / "profile_behavior_top_effects.csv",
        index=False,
    )

    _plot_top_effects(top_effect_df, top_pairs_df, args.output_dir / "profile_behavior_top_effects.png")

    manifest = {
        "runs": [run_name for run_name, _ in RUN_SPECS],
        "top_k": args.top_k,
        "cue_names": list(CUE_LABELS.keys()),
        "outcomes": [outcome_name for outcome_name, _, _ in OUTCOME_SPECS],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
