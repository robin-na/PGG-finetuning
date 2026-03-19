#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_COLORS: Dict[str, str] = {
    "human": "#222222",
    "oracle archetype": "#d97706",
    "retrieved archetype": "#b91c1c",
    "no archetype": "#1d4ed8",
    "random archetype": "#0f766e",
}

CATEGORY_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("coordination", "Coordination", r"\b(?:all|everyone|together|agree|agreed|contribute|contributing|put in|20|full|max|all in|cooperate|cooperat|fair|team|same page|commit)\b"),
    ("mentions_20_or_full", "20 / Full / Max", r"\b(?:20|full|all[- ]?in|max(?:imum)?|fully)\b"),
    ("meta_experiment", "Meta / Platform", r"\b(?:bot|bots|automated|human|prolific|bonus|payment|paid|coins|conversion|dollar|dollars|hour|hours|rounds|experiment|message|messages|typing|chat|idle)\b"),
    ("non_game_banter", "Banter / Identity", r"\b(?:uk|usa|england|london|friday|owl|snake|bird|sloth|duck|frog|moo|ribbit|lol|haha)\b"),
)


def split_csv_arg(value: str | None) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_chat_log(raw: object) -> List[Dict[str, object]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(f"[{text}]")
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(float) != 0.0
    true_values = {"1", "true", "t", "yes", "y"}
    return (
        series.astype(str).str.strip().str.lower().map(lambda x: x in true_values).fillna(False)
    )


def load_human_messages(analysis_csv: Path, game_ids: Iterable[str]) -> pd.DataFrame:
    analysis = pd.read_csv(analysis_csv)
    analysis["gameId"] = analysis["gameId"].astype(str)
    sub = analysis[analysis["gameId"].isin(set(map(str, game_ids)))].copy()
    rows: List[Dict[str, object]] = []
    for row in sub[["gameId", "chat_log"]].itertuples(index=False):
        for msg in parse_chat_log(row.chat_log):
            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            game_phase = str(msg.get("gamePhase", ""))
            round_match = re.search(r"Round\s+(\d+)", game_phase)
            round_index = int(round_match.group(1)) + 1 if round_match else np.nan
            phase = game_phase.split(" - ", 1)[1].strip().lower() if " - " in game_phase else ""
            rows.append(
                {
                    "source": "human",
                    "gameId": str(row.gameId),
                    "playerId": str(msg.get("playerId", "")),
                    "roundIndex": round_index,
                    "phase": phase,
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


def load_sim_messages(eval_csv: Path, label: str, game_ids: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(eval_csv)
    df["gameId"] = df["gameId"].astype(str)
    df["playerId"] = df["playerId"].astype(str)
    df = df[df["gameId"].isin(set(map(str, game_ids)))].copy()
    df["text"] = df["data.chat_message"].astype(str).str.strip()
    df = df[df["data.chat_message"].notna()].copy()
    df = df[df["text"] != ""].copy()
    df = df[df["text"].str.lower() != "nan"].copy()
    out = df[["gameId", "playerId", "roundIndex", "text"]].copy()
    out["source"] = label
    out["phase"] = "contribution_slot"
    return out


def compute_message_stats(messages: pd.DataFrame, game_meta: pd.DataFrame) -> pd.DataFrame:
    chat_games = set(game_meta["gameId"].astype(str))
    rounds_map = game_meta.set_index("gameId")["CONFIG_numRounds"].astype(float).to_dict()
    player_map = game_meta.set_index("gameId")["CONFIG_playerCount"].astype(float).to_dict()
    rows: List[Dict[str, object]] = []
    for source, sub in messages.groupby("source", sort=False):
        total_rounds = sum(float(rounds_map[g]) for g in chat_games if g in rounds_map)
        by_gr = sub.groupby(["gameId", "roundIndex"])["playerId"].nunique().reset_index(name="n_speakers")
        by_gr["player_count"] = by_gr["gameId"].map(player_map)
        by_gr["speaker_share"] = by_gr["n_speakers"] / by_gr["player_count"]
        by_gp = sub.groupby(["gameId", "playerId"]).size()
        words = sub["text"].astype(str).str.findall(r"[A-Za-z']+").map(len)
        chars = sub["text"].astype(str).str.len()
        rows.append(
            {
                "source": source,
                "active_games": int(sub["gameId"].nunique()),
                "total_messages": int(len(sub)),
                "messages_per_chat_enabled_game": float(len(sub) / len(chat_games)) if chat_games else np.nan,
                "messages_per_game_round": float(len(sub) / total_rounds) if total_rounds else np.nan,
                "median_unique_speakers_per_game_round": float(by_gr["n_speakers"].median()) if not by_gr.empty else np.nan,
                "mean_speaker_share_per_round": float(by_gr["speaker_share"].mean()) if not by_gr.empty else np.nan,
                "mean_messages_per_player_game": float(by_gp.mean()) if len(by_gp) else np.nan,
                "mean_words_per_message": float(words.mean()) if len(words) else np.nan,
                "mean_chars_per_message": float(chars.mean()) if len(chars) else np.nan,
                "unique_message_share": float(sub["text"].nunique() / len(sub)) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_category_rates(messages: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source, sub in messages.groupby("source", sort=False):
        lower = sub["text"].astype(str).str.lower().fillna("")
        for key, title, pattern in CATEGORY_SPECS:
            rows.append(
                {
                    "source": source,
                    "category": key,
                    "category_title": title,
                    "rate": float(lower.str.contains(pattern, regex=True).mean()) if len(lower) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def pick_examples(messages: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for source, sub in messages.groupby("source", sort=False):
        unique_text = sub["text"].drop_duplicates().tolist()
        examples: List[str] = []
        if source == "human":
            patterns = [
                r"\b(?:agree|max|contribute|work together|defaults to your max)\b",
                r"\b(?:lol|uk|usa|bonus|owl|sports|human|bot)\b",
            ]
            for pattern in patterns:
                for text in unique_text:
                    if re.search(pattern, text.lower()):
                        examples.append(text)
                        break
        else:
            vc = sub["text"].value_counts()
            if not vc.empty:
                examples.append(str(vc.index[0]))
            for text in unique_text:
                if "punish" in text.lower() or "reward" in text.lower() or "20" in text.lower():
                    if text not in examples:
                        examples.append(text)
                        break
        out[source] = examples[:2]
    return out


def build_takeaways(stats_df: pd.DataFrame, category_df: pd.DataFrame, phase_df: pd.DataFrame, labels: Sequence[str]) -> List[str]:
    stats = stats_df.set_index("source")
    cats = category_df.pivot(index="source", columns="category", values="rate").fillna(0.0)
    human = stats.loc["human"]
    bullets = []
    if labels:
        for label in labels:
            if label not in stats.index:
                continue
            row = stats.loc[label]
            bullets.append(
                f"{label}: {row['messages_per_game_round']:.2f} msgs/round vs human {human['messages_per_game_round']:.2f}; "
                f"speaker share {row['mean_speaker_share_per_round']:.2f} vs {human['mean_speaker_share_per_round']:.2f}."
            )
    if "human" in cats.index:
        bullets.append(
            "Humans use chat reactively: "
            + ", ".join(
                f"{phase} {100 * share:.0f}%"
                for phase, share in phase_df.set_index("phase")["share"].to_dict().items()
            )
            + ". Model chat is a single pre-action slot."
        )
        if labels:
            model_unique = [stats.loc[label, "unique_message_share"] for label in labels if label in stats.index]
            if model_unique:
                bullets.append(
                    f"Human chat is less templated ({100 * human['unique_message_share']:.0f}% unique messages) "
                    f"than the model runs ({' / '.join(f'{100 * value:.0f}%' for value in model_unique)})."
                )
    if len(labels) >= 2 and labels[0] in cats.index and labels[1] in cats.index:
        a = cats.loc[labels[0]]
        b = cats.loc[labels[1]]
        bullets.append(
            f"{labels[1]} is more locked into the full-contribution script "
            f"({100*b['mentions_20_or_full']:.0f}% mention 20/full/max vs {100*a['mentions_20_or_full']:.0f}% for {labels[0]})."
        )
    if "human" in cats.index and labels:
        last = labels[-1]
        if last in cats.index:
            bullets.append(
                f"Human chat has more non-game banter/identity talk ({100*cats.loc['human','non_game_banter']:.0f}%) "
                f"than {last} ({100*cats.loc[last,'non_game_banter']:.0f}%)."
            )
    return bullets[:4]


def _setup_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")


def make_summary_figure(
    stats_df: pd.DataFrame,
    category_df: pd.DataFrame,
    examples: Dict[str, List[str]],
    takeaways: Sequence[str],
    out_path: Path,
    title: str,
    subtitle: str,
) -> None:
    _setup_matplotlib()
    import matplotlib.pyplot as plt

    sources = list(stats_df["source"])
    colors = [DEFAULT_COLORS.get(source, "#666666") for source in sources]

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.1, 1.2])
    ax_table = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    ax_table.axis("off")
    table_metrics = [
        ("messages_per_game_round", "Msgs / Game-Round"),
        ("mean_speaker_share_per_round", "Speaker Share / Round"),
        ("mean_words_per_message", "Words / Msg"),
        ("unique_message_share", "Unique Msg Share"),
    ]
    table_rows = []
    row_labels = []
    for source in sources:
        row_labels.append(source.title())
        row = stats_df[stats_df["source"] == source].iloc[0]
        table_rows.append(
            [
                f"{float(row[col]):.2f}" if col != "unique_message_share" else f"{100 * float(row[col]):.0f}%"
                for col, _ in table_metrics
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        rowLabels=row_labels,
        colLabels=[label for _, label in table_metrics],
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)
    ax_table.set_title("Frequency Snapshot", fontsize=14, pad=10)

    cat_order = [key for key, _, _ in CATEGORY_SPECS]
    cat_titles = {key: title for key, title, _ in CATEGORY_SPECS}
    x = np.arange(len(cat_order), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, num=len(sources))
    for idx, source in enumerate(sources):
        sub = category_df[category_df["source"] == source].set_index("category").reindex(cat_order)
        vals = pd.to_numeric(sub["rate"], errors="coerce").to_numpy(dtype=float) * 100.0
        ax_bar.bar(x + offsets[idx], vals, width=width, color=colors[idx], label=source.title(), alpha=0.9)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([cat_titles[key] for key in cat_order], rotation=18, ha="right")
    ax_bar.set_ylabel("% of Messages")
    ax_bar.set_title("Content Mix", fontsize=14, pad=10)
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(loc="upper right")

    ax_text.axis("off")
    left = 0.02
    ax_text.text(left, 0.97, "What Stands Out", fontsize=15, fontweight="bold", va="top")
    bullet_lines = []
    for bullet in takeaways:
        wrapped = textwrap.fill(bullet, width=90, subsequent_indent="  ")
        bullet_lines.append(f"- {wrapped}")
    ax_text.text(left, 0.84, "\n".join(bullet_lines), fontsize=10.8, va="top", linespacing=1.45)

    box_y = 0.26
    box_w = 0.30
    gap = 0.03
    for idx, source in enumerate(sources):
        x0 = left + idx * (box_w + gap)
        if idx == 2:
            x0 = 0.68
        example_text = "\n".join(f'"{textwrap.shorten(msg, width=90, placeholder="...")}"' for msg in examples.get(source, []))
        ax_text.text(
            x0,
            box_y,
            f"{source.title()}\n\n{example_text or 'No example available.'}",
            fontsize=10.0,
            va="top",
            ha="left",
            bbox={"facecolor": "#f8f8f8", "edgecolor": DEFAULT_COLORS.get(source, "#999999"), "boxstyle": "round,pad=0.5"},
        )

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=11, color="#444444")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a one-page chat comparison summary.")
    parser.add_argument("--analysis_csv", type=Path, default=Path("benchmark/data/processed_data/df_analysis_val.csv"))
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("outputs/benchmark/runs/benchmark_filtered/macro_simulation_eval"),
    )
    parser.add_argument("--run_ids", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument(
        "--analysis_root",
        type=Path,
        default=Path("reports/benchmark/macro_simulation_eval"),
    )
    parser.add_argument("--analysis_run_id", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_ids = split_csv_arg(args.run_ids)
    labels = split_csv_arg(args.labels)
    if len(run_ids) != len(labels):
        raise ValueError("--labels must align with --run_ids.")

    analysis_csv = args.analysis_csv.resolve()
    eval_root = args.eval_root.resolve()
    out_dir = (args.analysis_root.resolve() / args.analysis_run_id).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    analysis = pd.read_csv(analysis_csv)
    analysis["gameId"] = analysis["gameId"].astype(str)
    shared_games = set(analysis["gameId"].astype(str))
    for run_id in run_ids:
        eval_csv = (eval_root / run_id / "macro_simulation_eval.csv").resolve()
        df = pd.read_csv(eval_csv, usecols=["gameId"])
        shared_games &= set(df["gameId"].astype(str))

    chat_meta = analysis[analysis["gameId"].isin(shared_games)].copy()
    chat_meta = chat_meta.loc[parse_bool_series(chat_meta["CONFIG_chat"])].copy()
    chat_meta = chat_meta[["gameId", "CONFIG_numRounds", "CONFIG_playerCount", "CONFIG_treatmentName"]].drop_duplicates("gameId")
    chat_game_ids = sorted(chat_meta["gameId"].astype(str))

    human_df = load_human_messages(analysis_csv, game_ids=chat_game_ids)
    frames = [human_df]
    for run_id, label in zip(run_ids, labels):
        eval_csv = (eval_root / run_id / "macro_simulation_eval.csv").resolve()
        frames.append(load_sim_messages(eval_csv, label=label, game_ids=chat_game_ids))
    messages = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    stats_df = compute_message_stats(messages, chat_meta)
    category_df = compute_category_rates(messages)
    examples = pick_examples(messages)
    human_phase = (
        human_df["phase"].value_counts(normalize=True).rename_axis("phase").reset_index(name="share")
        if not human_df.empty
        else pd.DataFrame(columns=["phase", "share"])
    )
    takeaways = build_takeaways(stats_df, category_df, human_phase, labels=labels)

    stats_df.to_csv(out_dir / "chat_frequency_summary.csv", index=False)
    category_df.to_csv(out_dir / "chat_category_rates.csv", index=False)
    pd.DataFrame(
        [{"source": source, "example_index": idx + 1, "text": text} for source, texts in examples.items() for idx, text in enumerate(texts)]
    ).to_csv(out_dir / "chat_examples.csv", index=False)

    summary_md = out_dir / "team_share_summary.md"
    md_lines = [
        f"# Chat Comparison Summary: Human vs {', '.join(labels)}",
        "",
        f"- Scope: `{len(chat_game_ids)}` chat-enabled shared games from `{analysis_csv}`.",
        "",
        "## Key Takeaways",
    ]
    md_lines.extend([f"- {line}" for line in takeaways])
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    title = f"Communication Patterns: Human vs {' vs '.join(labels)}"
    subtitle = f"{len(chat_game_ids)} shared chat-enabled games | benchmark validation split"
    make_summary_figure(
        stats_df=stats_df,
        category_df=category_df,
        examples=examples,
        takeaways=takeaways,
        out_path=fig_dir / "chat_comparison_summary.png",
        title=title,
        subtitle=subtitle,
    )

    manifest = {
        "analysis_csv": str(analysis_csv),
        "run_ids": run_ids,
        "labels": labels,
        "n_chat_enabled_shared_games": int(len(chat_game_ids)),
        "outputs": {
            "chat_frequency_summary_csv": str(out_dir / "chat_frequency_summary.csv"),
            "chat_category_rates_csv": str(out_dir / "chat_category_rates.csv"),
            "chat_examples_csv": str(out_dir / "chat_examples.csv"),
            "team_share_summary_md": str(summary_md),
            "chat_comparison_summary_png": str(fig_dir / "chat_comparison_summary.png"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote chat comparison summary -> {out_dir}")
    print(stats_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
