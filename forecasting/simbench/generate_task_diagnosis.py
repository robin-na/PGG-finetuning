from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib.pyplot as plt


DEFAULT_BASELINE_RUN = (
    "simbenchpop__baseline_group_batched_explained__gpt_5_mini__datasets_"
    "opinionqa_choices13k_ospsychbig5_ospsychrwas_dices__us_only"
)
DEFAULT_TWIN_RUN = (
    "simbenchpop__twin_profile_batched_seed_0__n64__gpt_5_mini__datasets_"
    "opinionqa_choices13k_ospsychbig5_ospsychrwas_dices__us_only"
)

TASK_NOTES = {
    "Choices13k": (
        "Raw Twin heuristic uses direct Twin lottery and time-risk tasks only. "
        "This isolates the source risk signal without an LLM in the loop."
    ),
    "DICES": (
        "Raw Twin heuristic is not applicable: Twin does not include a directly "
        "comparable safety-annotation task."
    ),
    "OSPsychBig5": (
        "Raw Twin heuristic uses raw Twin Big Five scores mapped to item polarity. "
        "It is approximate and intentionally simple, but it uses no LLM."
    ),
    "OSPsychRWAS": (
        "Raw Twin heuristic is not applicable: Twin has ideology/religion proxies, "
        "but no directly comparable RWAS measurement."
    ),
    "OpinionQA": (
        "Raw Twin heuristic is not applicable: Twin has relevant demographics and "
        "attitudes, but no directly comparable per-question survey responses. "
        "Also note that option letters are heterogeneous across OpinionQA questions, "
        "so A/B/C/... aggregation is only a coarse shape comparison."
    ),
}

TASK_INTERPRETATIONS = {
    "Choices13k": (
        "Twin raw risk signals retain much more question-to-question structure than "
        "the current Twin+LLM pipeline. The current persona prompting appears to "
        "wash out gamble sensitivity and overpredict one stable gamble preference."
    ),
    "DICES": (
        "Twin LLM shifts mass toward unsafe and unsure relative to the human labels, "
        "which is consistent with Twin injecting a broad moral alarm signal instead "
        "of the narrow safety taxonomy used in DICES."
    ),
    "OSPsychBig5": (
        "Twin raw Big Five signals are more faithful than Twin+LLM, but the current "
        "Twin cards still over-idealize the person and push responses away from the "
        "middle, especially on negative-keyed items."
    ),
    "OSPsychRWAS": (
        "Twin improves on this task despite lacking a direct raw RWAS signal, which "
        "suggests the value is coming from broad ideology/religion proxies rather "
        "than exact item-level transport."
    ),
    "OpinionQA": (
        "At the coarse task level, baseline and Twin are both close to the human "
        "option-position distribution. Residual errors here are more local and "
        "question-specific than task-wide."
    ),
}

BIG5_ITEM_MAPPING = {
    "I am always prepared.": ("big_five_conscientiousness", True),
    "I am easily disturbed.": ("big_five_neuroticism", True),
    "I am full of ideas.": ("big_five_openness", True),
    "I am interested in people.": ("big_five_agreeableness", True),
    "I am not interested in other people's problems.": ("big_five_agreeableness", False),
    "I am not really interested in others.": ("big_five_agreeableness", False),
    "I am quick to understand things.": ("big_five_openness", True),
    "I am quiet around strangers.": ("big_five_extraversion", False),
    "I am relaxed most of the time.": ("big_five_neuroticism", False),
    "I am the life of the party.": ("big_five_extraversion", True),
    "I change my mood a lot.": ("big_five_neuroticism", True),
    "I do not have a good imagination.": ("big_five_openness", False),
    "I don't like to draw attention to myself.": ("big_five_extraversion", False),
    "I don't mind being the center of attention.": ("big_five_extraversion", True),
    "I don't talk a lot.": ("big_five_extraversion", False),
    "I feel comfortable around people.": ("big_five_extraversion", True),
    "I feel little concern for others.": ("big_five_agreeableness", False),
    "I follow a schedule.": ("big_five_conscientiousness", True),
    "I get chores done right away.": ("big_five_conscientiousness", True),
    "I get irritated easily.": ("big_five_neuroticism", True),
    "I get upset easily.": ("big_five_neuroticism", True),
    "I have a rich vocabulary.": ("big_five_openness", True),
    "I have a vivid imagination.": ("big_five_openness", True),
    "I have excellent ideas.": ("big_five_openness", True),
    "I have frequent mood swings.": ("big_five_neuroticism", True),
    "I have little to say.": ("big_five_extraversion", False),
    "I keep in the background.": ("big_five_extraversion", False),
    "I like order.": ("big_five_conscientiousness", True),
    "I make a mess of things.": ("big_five_conscientiousness", False),
    "I make people feel at ease.": ("big_five_agreeableness", True),
    "I often feel blue.": ("big_five_neuroticism", True),
    "I often forget to put things back in their proper place.": (
        "big_five_conscientiousness",
        False,
    ),
    "I pay attention to details.": ("big_five_conscientiousness", True),
    "I seldom feel blue.": ("big_five_neuroticism", False),
    "I spend time reflecting on things.": ("big_five_openness", True),
    "I start conversations.": ("big_five_extraversion", True),
    "I sympathize with others' feelings.": ("big_five_agreeableness", True),
    "I take time out for others.": ("big_five_agreeableness", True),
    "I talk to a lot of different people at parties.": ("big_five_extraversion", True),
    "I use difficult words.": ("big_five_openness", True),
}


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _parse_distribution(text: str) -> dict[str, float]:
    if pd.isna(text) or not str(text).strip():
        return {}
    parsed = json.loads(text)
    return {str(k): float(v) for k, v in parsed.items()}


def _mean_distribution(dist_series: pd.Series) -> dict[str, float]:
    distributions = [_parse_distribution(value) for value in dist_series]
    labels = sorted({label for dist in distributions for label in dist})
    if not labels:
        return {}
    return {
        label: float(sum(dist.get(label, 0.0) for dist in distributions) / len(distributions))
        for label in labels
    }


def _tvd(pred: dict[str, float], gold: dict[str, float]) -> float:
    labels = sorted(set(pred) | set(gold))
    return 0.5 * sum(abs(pred.get(label, 0.0) - gold.get(label, 0.0)) for label in labels)


def _load_row_eval(results_root: Path, run_name: str) -> pd.DataFrame:
    path = results_root / f"{run_name}__gold_eval" / "row_level_evaluation.csv"
    df = pd.read_csv(path)
    if "evaluated" in df.columns:
        df = df[df["evaluated"].astype(bool)].copy()
    return df


def _load_selected_rows(metadata_root: Path, run_name: str) -> pd.DataFrame:
    path = metadata_root / run_name / "selected_rows.csv"
    return pd.read_csv(path)


def _compute_choices13k_raw_heuristic(selected_rows: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    profiles_path = (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profiles"
        / "twin_extended_profiles.jsonl"
    )

    twin_rows: list[dict[str, float]] = []
    with profiles_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            profile = json.loads(line)
            sec = ((profile.get("observed_in_twin") or {}).get("economic_preferences_non_social") or {})
            feats = {
                item["name"]: item["value"]["raw"]
                for item in sec.get("summary_features", [])
                if isinstance(item, dict)
                and "name" in item
                and isinstance(item.get("value"), dict)
                and "raw" in item["value"]
            }
            gain_rate = feats.get("lottery_choice_rate_gains")
            loss_rate = feats.get("lottery_choice_rate_losses")
            if gain_rate is None or loss_rate is None:
                continue
            twin_rows.append(
                {
                    "gain_rate": float(gain_rate),
                    "loss_rate": float(loss_rate),
                }
            )
    profiles = pd.DataFrame(twin_rows)

    gain_tasks = [
        {"lottery": [(0.5, 6.0), (0.5, 0.0)], "certs": [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0]},
        {"lottery": [(0.5, 8.0), (0.5, 2.0)], "certs": [2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 6.0, 7.0]},
        {"lottery": [(0.5, 10.0), (0.5, 0.0)], "certs": [2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 6.0, 7.0]},
    ]
    loss_tasks = [
        {"lottery": [(0.5, -6.0), (0.5, 0.0)], "certs": [-5.5, -5.0, -4.75, -4.5, -4.25, -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.0, -1.0]},
        {"lottery": [(0.5, -8.0), (0.5, -2.0)], "certs": [-7.5, -7.0, -6.75, -6.5, -6.25, -6.0, -5.75, -5.5, -5.25, -5.0, -4.75, -4.5, -4.0, -3.0]},
        {"lottery": [(0.5, -10.0), (0.5, 0.0)], "certs": [-7.5, -7.0, -6.75, -6.5, -6.25, -6.0, -5.75, -5.5, -5.25, -5.0, -4.75, -4.5, -4.0, -3.0]},
        *(
            {"lottery": [(0.5, -8.0), (0.5, x)], "certs": [0.0]}
            for x in [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        ),
    ]

    def value_fn(x: float, rho_gain: float, rho_loss: float) -> float:
        if x >= 0:
            return 0.0 if x == 0 else x**rho_gain
        return -((-x) ** rho_loss)

    def avg_choice_rate_gain(rho_gain: float) -> float:
        total = 0
        chosen = 0.0
        for task in gain_tasks:
            eu = sum(prob * value_fn(payoff, rho_gain, 1.0) for prob, payoff in task["lottery"])
            for cert in task["certs"]:
                uc = value_fn(cert, rho_gain, 1.0)
                if abs(eu - uc) < 1e-12:
                    chosen += 0.5
                elif eu > uc:
                    chosen += 1.0
                total += 1
        return chosen / total

    def avg_choice_rate_loss(rho_loss: float, rho_gain: float) -> float:
        total = 0
        chosen = 0.0
        for task in loss_tasks:
            eu = sum(prob * value_fn(payoff, rho_gain, rho_loss) for prob, payoff in task["lottery"])
            for cert in task["certs"]:
                uc = value_fn(cert, rho_gain, rho_loss)
                if abs(eu - uc) < 1e-12:
                    chosen += 0.5
                elif eu > uc:
                    chosen += 1.0
                total += 1
        return chosen / total

    rho_grid = np.linspace(0.2, 2.5, 461)
    gain_rate_grid = np.array([avg_choice_rate_gain(rho) for rho in rho_grid])

    def fit_rho_gain(target: float) -> float:
        idx = int(np.argmin(np.abs(gain_rate_grid - target)))
        return float(rho_grid[idx])

    def fit_rho_loss(target: float, rho_gain: float) -> float:
        loss_rate_grid = np.array([avg_choice_rate_loss(rho, rho_gain) for rho in rho_grid])
        idx = int(np.argmin(np.abs(loss_rate_grid - target)))
        return float(rho_grid[idx])

    fitted = []
    for row in profiles.itertuples(index=False):
        rho_gain = fit_rho_gain(row.gain_rate)
        rho_loss = fit_rho_loss(row.loss_rate, rho_gain)
        fitted.append({"rho_gain": rho_gain, "rho_loss": rho_loss})
    fit_df = pd.DataFrame(fitted)

    machine_re = re.compile(r"Machine ([AB]): (.*?)(?=\nMachine [AB]:|\nWhich machine do you choose\?|$)", re.S)
    outcome_re = re.compile(r"\$(-?\d+(?:\.\d+)?) with ([0-9.]+)% chance")

    def parse_question(body: str) -> dict[str, list[tuple[float, float]]]:
        machines: dict[str, list[tuple[float, float]]] = {}
        for label, text in machine_re.findall(body):
            outcomes: list[tuple[float, float]] = []
            for value, prob in outcome_re.findall(text):
                outcomes.append((float(prob) / 100.0, float(value)))
            machines[label] = outcomes
        return machines

    def eu(outcomes: list[tuple[float, float]], rho_gain: float, rho_loss: float) -> float:
        return sum(prob * value_fn(payoff, rho_gain, rho_loss) for prob, payoff in outcomes)

    predictions = []
    for row in selected_rows.itertuples(index=False):
        machines = parse_question(row.question_body)
        choose_a = []
        for prof in fit_df.itertuples(index=False):
            ua = eu(machines["A"], prof.rho_gain, prof.rho_loss)
            ub = eu(machines["B"], prof.rho_gain, prof.rho_loss)
            if abs(ua - ub) < 1e-12:
                choose_a.append(0.5)
            else:
                choose_a.append(1.0 if ua > ub else 0.0)
        pred_a = float(np.mean(choose_a))
        predictions.append(
            {
                "simbench_row_id": row.simbench_row_id,
                "predicted_distribution_json": json.dumps({"A": pred_a, "B": 1.0 - pred_a}),
            }
        )
    return pd.DataFrame(predictions)


def _compute_ospsychbig5_raw_heuristic(selected_rows: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    profiles_path = (
        repo_root
        / "non-PGG_generalization"
        / "twin_profiles"
        / "output"
        / "twin_extended_profiles"
        / "twin_extended_profiles.jsonl"
    )

    rows: list[dict[str, float]] = []
    with profiles_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            profile = json.loads(line)
            sec = ((profile.get("observed_in_twin") or {}).get("personality_and_self_report") or {})
            feats = {
                item["name"]: item["value"]["raw"]
                for item in sec.get("summary_features", [])
                if isinstance(item, dict)
                and "name" in item
                and isinstance(item.get("value"), dict)
                and "raw" in item["value"]
            }
            keep = {
                "big_five_extraversion": feats.get("big_five_extraversion"),
                "big_five_agreeableness": feats.get("big_five_agreeableness"),
                "big_five_conscientiousness": feats.get("big_five_conscientiousness"),
                "big_five_neuroticism": feats.get("big_five_neuroticism"),
                "big_five_openness": feats.get("big_five_openness"),
            }
            if all(value is not None for value in keep.values()):
                rows.append({key: float(value) for key, value in keep.items()})
    profiles = pd.DataFrame(rows)

    def extract_statement(question_body: str) -> str:
        marker = "statement: "
        return question_body.split(marker, 1)[1].strip()

    def dist_from_score(score: float, positive: bool, sigma: float = 0.8) -> dict[str, float]:
        mu = 1.0 + 4.0 * (score / 100.0)
        if not positive:
            mu = 6.0 - mu
        xs = np.arange(1, 6)
        probs = np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        probs = probs / probs.sum()
        return {label: float(prob) for label, prob in zip(["A", "B", "C", "D", "E"], probs)}

    predictions = []
    for row in selected_rows.itertuples(index=False):
        statement = extract_statement(row.question_body)
        if statement not in BIG5_ITEM_MAPPING:
            raise KeyError(f"Missing OSPsychBig5 mapping for statement: {statement}")
        trait_name, positive = BIG5_ITEM_MAPPING[statement]
        panel = [dist_from_score(score, positive) for score in profiles[trait_name]]
        aggregated = {
            label: float(sum(dist[label] for dist in panel) / len(panel))
            for label in ["A", "B", "C", "D", "E"]
        }
        predictions.append(
            {
                "simbench_row_id": row.simbench_row_id,
                "predicted_distribution_json": json.dumps(aggregated),
            }
        )
    return pd.DataFrame(predictions)


def _compute_raw_predictions(task_name: str, selected_rows: pd.DataFrame, repo_root: Path) -> pd.DataFrame | None:
    if task_name == "Choices13k":
        return _compute_choices13k_raw_heuristic(selected_rows, repo_root)
    if task_name == "OSPsychBig5":
        return _compute_ospsychbig5_raw_heuristic(selected_rows, repo_root)
    return None


def _plot_task(
    *,
    task_name: str,
    output_path: Path,
    human_dist: dict[str, float],
    baseline_dist: dict[str, float],
    twin_dist: dict[str, float],
    raw_dist: dict[str, float] | None,
    baseline_tvd: float,
    twin_tvd: float,
    raw_tvd: float | None,
    note: str,
) -> None:
    labels = sorted(set(human_dist) | set(baseline_dist) | set(twin_dist) | set(raw_dist or {}))
    x = np.arange(len(labels))
    series = [
        ("Human", human_dist, "#222222"),
        ("Base LLM", baseline_dist, "#1f77b4"),
        ("Twin LLM", twin_dist, "#d62728"),
    ]
    if raw_dist is not None:
        series.append(("Twin Raw", raw_dist, "#2ca02c"))

    if len(series) == 4:
        offset_step = 0.16
        width = 0.28
    else:
        offset_step = 0.18
        width = 0.30
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * offset_step

    fig, ax = plt.subplots(figsize=(11, 6))
    for zorder, (offset, (name, dist, color)) in enumerate(zip(offsets, series), start=1):
        values = [dist.get(label, 0.0) for label in labels]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            alpha=0.62,
            label=name,
            edgecolor="none",
            zorder=zorder,
        )

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Option Mass")
    ax.set_title(task_name)
    ax.legend(loc="upper right", frameon=False)

    metric_lines = [
        f"Base LLM mean TVD: {baseline_tvd:.3f}",
        f"Twin LLM mean TVD: {twin_tvd:.3f}",
    ]
    if raw_tvd is None:
        metric_lines.append("Twin Raw: N/A")
    else:
        metric_lines.append(f"Twin Raw mean TVD: {raw_tvd:.3f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(metric_lines),
        ha="left",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary(
    *,
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "task_summary.csv", index=False)

    lines = ["# SimBench Task Diagnosis", ""]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['task_name']}",
                "",
                f"- Rows compared: {row['n_overlap_rows']}",
                f"- Mean TVD: baseline `{row['baseline_mean_tvd']:.4f}`, Twin `{row['twin_mean_tvd']:.4f}`"
                + (
                    f", Twin raw `{row['raw_mean_tvd']:.4f}`"
                    if row["raw_mean_tvd"] == row["raw_mean_tvd"]
                    else ", Twin raw `N/A`"
                ),
                f"- Plot: [{row['plot_file']}](./{row['plot_file']})",
                f"- Note: {TASK_NOTES.get(row['task_name'], '')}",
                f"- Diagnosis: {TASK_INTERPRETATIONS.get(row['task_name'], '')}",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_task_diagnosis(
    *,
    repo_root: Path,
    baseline_run: str,
    twin_run: str,
    output_dir: Path,
) -> None:
    results_root = repo_root / "forecasting" / "simbench" / "results"
    metadata_root = repo_root / "forecasting" / "simbench" / "metadata"

    baseline_eval = _load_row_eval(results_root, baseline_run)
    twin_eval = _load_row_eval(results_root, twin_run)
    selected_rows = _load_selected_rows(metadata_root, twin_run)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_heuristics"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    common_tasks = sorted(set(baseline_eval["dataset_name"]) & set(twin_eval["dataset_name"]))
    for task_name in common_tasks:
        base_task = baseline_eval[baseline_eval["dataset_name"] == task_name].copy()
        twin_task = twin_eval[twin_eval["dataset_name"] == task_name].copy()
        overlap_row_ids = sorted(set(base_task["simbench_row_id"]) & set(twin_task["simbench_row_id"]))
        if not overlap_row_ids:
            continue

        base_task = base_task[base_task["simbench_row_id"].isin(overlap_row_ids)].copy()
        twin_task = twin_task[twin_task["simbench_row_id"].isin(overlap_row_ids)].copy()
        task_rows = selected_rows[
            (selected_rows["dataset_name"] == task_name)
            & (selected_rows["simbench_row_id"].isin(overlap_row_ids))
        ].copy()

        human_dist = _mean_distribution(twin_task["gold_distribution_json"])
        baseline_dist = _mean_distribution(base_task["predicted_distribution_json"])
        twin_dist = _mean_distribution(twin_task["predicted_distribution_json"])

        baseline_mean_tvd = float(base_task["tvd"].mean())
        twin_mean_tvd = float(twin_task["tvd"].mean())

        raw_predictions = _compute_raw_predictions(task_name, task_rows, repo_root)
        raw_dist: dict[str, float] | None = None
        raw_mean_tvd: float | None = None
        if raw_predictions is not None:
            raw_eval = task_rows[["simbench_row_id", "gold_distribution_json"]].merge(
                raw_predictions,
                on="simbench_row_id",
                how="inner",
            )
            raw_eval["tvd"] = raw_eval.apply(
                lambda row: _tvd(
                    _parse_distribution(row["predicted_distribution_json"]),
                    _parse_distribution(row["gold_distribution_json"]),
                ),
                axis=1,
            )
            raw_dist = _mean_distribution(raw_eval["predicted_distribution_json"])
            raw_mean_tvd = float(raw_eval["tvd"].mean())
            raw_eval.to_csv(raw_dir / f"{_slugify(task_name)}_raw_eval.csv", index=False)

        plot_file = f"{_slugify(task_name)}_distribution_diagnosis.png"
        _plot_task(
            task_name=task_name,
            output_path=output_dir / plot_file,
            human_dist=human_dist,
            baseline_dist=baseline_dist,
            twin_dist=twin_dist,
            raw_dist=raw_dist,
            baseline_tvd=baseline_mean_tvd,
            twin_tvd=twin_mean_tvd,
            raw_tvd=raw_mean_tvd,
            note=TASK_NOTES.get(task_name, ""),
        )

        summary_rows.append(
            {
                "task_name": task_name,
                "n_overlap_rows": len(overlap_row_ids),
                "baseline_mean_tvd": baseline_mean_tvd,
                "twin_mean_tvd": twin_mean_tvd,
                "raw_mean_tvd": float("nan") if raw_mean_tvd is None else raw_mean_tvd,
                "raw_available": raw_predictions is not None,
                "gold_distribution_json": json.dumps(human_dist, sort_keys=True),
                "baseline_distribution_json": json.dumps(baseline_dist, sort_keys=True),
                "twin_distribution_json": json.dumps(twin_dist, sort_keys=True),
                "raw_distribution_json": "" if raw_dist is None else json.dumps(raw_dist, sort_keys=True),
                "plot_file": plot_file,
            }
        )

    _write_summary(output_dir=output_dir, summary_rows=summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SimBench task-level diagnosis plots.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--twin-run", default=DEFAULT_TWIN_RUN)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            args.repo_root
            / "forecasting"
            / "simbench"
            / "results"
            / (
                "simbenchpop__task_diagnosis__gpt_5_mini__datasets_"
                "opinionqa_choices13k_ospsychbig5_ospsychrwas_dices__us_only"
            )
        )
    build_task_diagnosis(
        repo_root=args.repo_root,
        baseline_run=args.baseline_run,
        twin_run=args.twin_run,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
