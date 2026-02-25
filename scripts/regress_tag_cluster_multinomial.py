#!/usr/bin/env python3
"""
Multinomial regression for cluster distribution by CONFIG values.

For each tag-specific clustered output:
- Fit multinomial logistic regression: cluster_id ~ CONFIG features.
- Report CV metrics.
- Export coefficients and feature effect summaries.
- Plot:
  1) coefficient heatmap (cluster x feature)
  2) feature effect bar chart
  3) MPCR marginal predicted distribution (stacked)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CLUSTER_ROOT = Path("Persona/misc/tag_section_clusters_openai")
DEFAULT_CONFIG_CSV = Path("data/processed_data/df_analysis_learn.csv")
DEFAULT_OUTPUT_DIR = Path("Persona/misc/tag_section_clusters_openai/analysis_regression_multinomial")

# Excluding: punishmentCost, punishmentTech, rewardCost, rewardTech (as requested)
DEFAULT_FEATURES = [
    "CONFIG_chat",
    "CONFIG_punishmentExists",
    "CONFIG_showRewardId",
    "CONFIG_rewardMagnitude",
    "CONFIG_punishmentMagnitude",
    "CONFIG_showNRounds",
    "CONFIG_endowment",
    "CONFIG_showPunishmentId",
    "CONFIG_rewardExists",
    "CONFIG_MPCR",
    "CONFIG_showOtherSummaries",
    "CONFIG_numRounds",
    "CONFIG_allOrNothing",
    "CONFIG_defaultContribProp",
    "CONFIG_playerCount",
]


def to_bool_num(v: Any) -> Any:
    if v is None:
        return np.nan
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().lower()
    if s in {"true", "t", "yes", "y"}:
        return 1.0
    if s in {"false", "f", "no", "n"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return np.nan


def load_cluster_rows(cluster_root: Path) -> pd.DataFrame:
    files = sorted(cluster_root.glob("*/*_clustered.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *_clustered.jsonl files found under {cluster_root}")
    rows: List[dict] = []
    for p in files:
        tag = p.parent.name
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(
                    {
                        "tag": tag,
                        "experiment": str(obj.get("experiment")),
                        "cluster_id": int(obj.get("cluster_id")),
                    }
                )
    return pd.DataFrame(rows)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=4000,
                    C=1.0,
                ),
            ),
        ]
    )


def choose_cv(y: np.ndarray) -> Optional[StratifiedKFold]:
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min()) if len(counts) else 0
    n_splits = min(5, min_count)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def get_feature_values_for_marginal(x: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    vals = x.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.array([]), np.array([])

    uniq = np.unique(vals)
    if len(uniq) <= 12:
        weights = np.array([(vals == v).mean() for v in uniq], dtype=float)
        return uniq, weights

    qs = np.linspace(0.05, 0.95, 10)
    qvals = np.unique(np.quantile(vals, qs))
    weights = np.ones(len(qvals), dtype=float) / max(1, len(qvals))
    return qvals, weights


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def compute_marginal_effects(
    model: Pipeline,
    x_df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    base_pred = model.predict_proba(x_df)
    p0 = base_pred.mean(axis=0)
    rows = []
    for col in features:
        vals, w = get_feature_values_for_marginal(x_df[col])
        if len(vals) == 0:
            rows.append({"feature": col, "marginal_tv": np.nan, "n_eval_values": 0})
            continue

        tvs = []
        for v in vals:
            xv = x_df.copy()
            xv[col] = v
            pv = model.predict_proba(xv).mean(axis=0)
            tvs.append(total_variation(pv, p0))
        tvs = np.asarray(tvs, dtype=float)
        marginal_tv = float(np.sum(w * tvs) / np.sum(w))
        rows.append({"feature": col, "marginal_tv": marginal_tv, "n_eval_values": int(len(vals))})
    return pd.DataFrame(rows)


def plot_coef_heatmap(
    coef: np.ndarray,
    classes: np.ndarray,
    features: List[str],
    out_path: Path,
    title: str,
) -> None:
    fig_w = max(10, 0.55 * len(features))
    fig_h = max(5, 0.40 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(coef))) if coef.size else 1.0
    vmax = max(vmax, 1e-9)
    im = ax.imshow(coef, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster class")
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels([str(int(c)) for c in classes], fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Standardized coefficient")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_feature_effects(effect_df: pd.DataFrame, out_path: Path, title: str) -> None:
    sub = effect_df.sort_values("marginal_tv", ascending=False).copy()
    fig_h = max(4, 0.35 * len(sub) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), constrained_layout=True)
    ax.barh(sub["feature"], sub["marginal_tv"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("Marginal TV shift in predicted cluster distribution")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_mpcr_marginal(
    model: Pipeline,
    x_df: pd.DataFrame,
    classes: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    mpcr = pd.to_numeric(x_df["CONFIG_MPCR"], errors="coerce")
    mpcr = mpcr.dropna()
    if mpcr.empty or mpcr.nunique() < 2:
        return

    q = np.linspace(0.05, 0.95, 10)
    vals = np.unique(np.quantile(mpcr.to_numpy(dtype=float), q))
    if len(vals) < 2:
        return

    preds = []
    for v in vals:
        xv = x_df.copy()
        xv["CONFIG_MPCR"] = float(v)
        p = model.predict_proba(xv).mean(axis=0)
        preds.append(p)
    mat = np.asarray(preds, dtype=float)  # [n_vals, n_classes]

    x = np.arange(len(vals))
    fig_w = max(10, 0.65 * len(vals))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("tab20")
    bottom = np.zeros(len(vals), dtype=float)
    for i, cls in enumerate(classes):
        y = mat[:, i]
        ax.bar(x, y, bottom=bottom, color=cmap(i % 20), width=0.88, label=f"{int(cls)}")
        bottom += y

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in vals], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("CONFIG_MPCR (quantile grid)")
    ax.set_ylabel("Predicted cluster proportion")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="cluster_id", bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=7)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multinomial regression for cluster distributions by CONFIG features")
    parser.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--config-csv", type=Path, default=DEFAULT_CONFIG_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES))
    args = parser.parse_args()

    cluster_df = load_cluster_rows(args.cluster_root)
    cfg = pd.read_csv(args.config_csv)
    cfg["gameId"] = cfg["gameId"].astype(str)

    need_cols = ["gameId"] + list(args.features)
    missing_cols = [c for c in need_cols if c not in cfg.columns]
    if missing_cols:
        raise ValueError(f"Missing CONFIG columns in {args.config_csv}: {missing_cols}")

    cfg = cfg[need_cols].drop_duplicates(subset=["gameId"], keep="first").copy()
    for col in args.features:
        cfg[col] = cfg[col].map(to_bool_num)

    merged = cluster_df.merge(cfg, how="left", left_on="experiment", right_on="gameId")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[dict] = []
    coef_rows: List[dict] = []
    effect_rows: List[dict] = []
    manifest: Dict[str, Any] = {
        "cluster_root": str(args.cluster_root),
        "config_csv": str(args.config_csv),
        "output_dir": str(args.output_dir),
        "n_rows_total": int(len(merged)),
        "n_unmatched_rows": int(merged["gameId"].isna().sum()),
        "features_requested": list(args.features),
        "tags": {},
    }

    for tag, sub in merged.groupby("tag"):
        tag_dir = args.output_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)

        y = sub["cluster_id"].to_numpy(dtype=int)
        x = sub[list(args.features)].copy()

        keep_features = []
        for col in x.columns:
            nunq = x[col].dropna().nunique()
            if nunq >= 2:
                keep_features.append(col)
        x = x[keep_features]

        if x.empty:
            manifest["tags"][tag] = {"status": "skipped_no_variable_features", "n_rows": int(len(sub))}
            continue

        model = build_model()

        cv = choose_cv(y)
        if cv is None:
            cv_acc = np.nan
            cv_ll = np.nan
            cv_splits = 0
        else:
            cv_splits = cv.get_n_splits()
            cv_acc = float(cross_val_score(model, x, y, cv=cv, scoring="accuracy").mean())
            cv_ll = float(-cross_val_score(model, x, y, cv=cv, scoring="neg_log_loss").mean())

        model.fit(x, y)
        y_pred = model.predict(x)
        y_prob = model.predict_proba(x)
        train_acc = float(accuracy_score(y, y_pred))
        train_ll = float(log_loss(y, y_prob, labels=np.unique(y)))

        clf: LogisticRegression = model.named_steps["clf"]
        coef = clf.coef_  # [n_classes, n_features]
        classes = clf.classes_

        metric_row = {
            "tag": tag,
            "n_rows": int(len(sub)),
            "n_clusters": int(len(np.unique(y))),
            "n_features_used": int(len(keep_features)),
            "cv_splits": int(cv_splits),
            "cv_accuracy_mean": cv_acc,
            "cv_log_loss_mean": cv_ll,
            "train_accuracy": train_acc,
            "train_log_loss": train_ll,
        }
        metrics_rows.append(metric_row)

        for i, cls in enumerate(classes):
            for j, feat in enumerate(keep_features):
                coef_rows.append(
                    {
                        "tag": tag,
                        "cluster_id": int(cls),
                        "feature": feat,
                        "coef": float(coef[i, j]),
                        "abs_coef": float(abs(coef[i, j])),
                    }
                )

        coef_abs_mean = np.mean(np.abs(coef), axis=0)
        coef_imp_df = pd.DataFrame({"feature": keep_features, "coef_abs_mean": coef_abs_mean})

        marginal_df = compute_marginal_effects(model, x, keep_features)
        merged_effect = marginal_df.merge(coef_imp_df, on="feature", how="left")
        merged_effect["tag"] = tag
        effect_rows.extend(merged_effect.to_dict(orient="records"))

        plot_coef_heatmap(
            coef=coef,
            classes=classes,
            features=keep_features,
            out_path=tag_dir / "coef_heatmap.png",
            title=f"{tag}: multinomial coefficients (cluster_id ~ CONFIG)",
        )
        plot_feature_effects(
            merged_effect[["feature", "marginal_tv"]].copy(),
            out_path=tag_dir / "feature_marginal_effects.png",
            title=f"{tag}: feature effect on predicted cluster distribution",
        )
        if "CONFIG_MPCR" in keep_features:
            plot_mpcr_marginal(
                model=model,
                x_df=x.copy(),
                classes=classes,
                out_path=tag_dir / "mpcr_marginal_predicted_distribution.png",
                title=f"{tag}: predicted cluster mix across MPCR",
            )

        merged_effect.sort_values("marginal_tv", ascending=False).to_csv(
            tag_dir / "feature_effects.csv",
            index=False,
        )

        manifest["tags"][tag] = {
            "status": "ok",
            "n_rows": int(len(sub)),
            "n_clusters": int(len(np.unique(y))),
            "n_features_used": int(len(keep_features)),
            "features_used": keep_features,
            "metrics": metric_row,
            "outputs": {
                "coef_heatmap_png": str(tag_dir / "coef_heatmap.png"),
                "feature_marginal_effects_png": str(tag_dir / "feature_marginal_effects.png"),
                "mpcr_marginal_predicted_distribution_png": str(tag_dir / "mpcr_marginal_predicted_distribution.png"),
                "feature_effects_csv": str(tag_dir / "feature_effects.csv"),
            },
        }

    metrics_df = pd.DataFrame(metrics_rows).sort_values("tag")
    coef_df = pd.DataFrame(coef_rows).sort_values(["tag", "cluster_id", "feature"])
    effects_df = pd.DataFrame(effect_rows).sort_values(["tag", "marginal_tv"], ascending=[True, False])

    metrics_df.to_csv(args.output_dir / "metrics_by_tag.csv", index=False)
    coef_df.to_csv(args.output_dir / "coefficients_long.csv", index=False)
    effects_df.to_csv(args.output_dir / "feature_effects_by_tag.csv", index=False)

    if not effects_df.empty:
        pivot = effects_df.pivot_table(index="tag", columns="feature", values="marginal_tv", aggfunc="mean").fillna(0.0)
        fig_w = max(12, 0.75 * len(pivot.columns))
        fig_h = max(4, 0.65 * len(pivot.index))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title("Feature marginal TV effects (tag x feature)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Marginal TV effect")
        fig.savefig(args.output_dir / "feature_effects_heatmap.png", dpi=220)
        plt.close(fig)

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved regression outputs to: {args.output_dir}")
    print(f"Tags processed: {len(manifest['tags'])}")
    print(f"Unmatched rows: {manifest['n_unmatched_rows']}")


if __name__ == "__main__":
    main()
