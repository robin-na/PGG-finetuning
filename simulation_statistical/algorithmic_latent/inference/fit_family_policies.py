from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = str(SCRIPT_PATH.parent)
ALGORITHMIC_LATENT_ROOT = SCRIPT_PATH.parents[1]
SIMULATION_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[3]
for path in (SCRIPT_DIR, str(SIMULATION_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from simulation_statistical.algorithmic_latent.inference.family_definitions import (  # noqa: E402
    FAMILY_LIBRARY,
    AlgorithmFamilySpec,
)
from simulation_statistical.algorithmic_latent.inference.build_state_table import (  # noqa: E402
    DEFAULT_STATE_TABLE_ROOT,
)


DEFAULT_MODEL_OUTPUT_PATH = ALGORITHMIC_LATENT_ROOT / "artifacts" / "models" / "family_policy_bundle.pkl"
DEFAULT_SUMMARY_OUTPUT_PATH = ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "family_policy_train_summary.csv"
DEFAULT_METADATA_OUTPUT_PATH = ALGORITHMIC_LATENT_ROOT / "artifacts" / "outputs" / "family_policy_train_summary.json"


def _unique_features(feature_names: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(feature) for feature in feature_names))


def _row_to_feature_dict(row: Mapping[str, Any], feature_names: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for feature in feature_names:
        value = row.get(feature)
        if value is None or pd.isna(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            out[feature] = int(bool(value))
        elif isinstance(value, (int, np.integer, float, np.floating)):
            out[feature] = float(value)
        else:
            out[feature] = str(value)
    return out


def _build_records(frame: pd.DataFrame, feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    subset = frame.loc[:, list(feature_names)].to_dict(orient="records")
    return [_row_to_feature_dict(record, feature_names) for record in subset]


def _build_classifier(*, class_weight: str | dict[str, float] | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True, sort=False)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=2000,
                    tol=1e-3,
                    early_stopping=True,
                    n_iter_no_change=5,
                    validation_fraction=0.05,
                    class_weight=class_weight,
                    random_state=0,
                ),
            ),
        ]
    )


def _fit_family_models(
    *,
    contribution_df: pd.DataFrame,
    action_df: pd.DataFrame,
    family_spec: AlgorithmFamilySpec,
) -> Dict[str, Any]:
    contribution_rows = contribution_df.dropna(subset=["actual_contribution_bin5"]).copy()
    action_rows = action_df.dropna(subset=["observed_action_label"]).copy()

    contribution_features = _unique_features(family_spec.contribution_features)
    action_features = _unique_features(family_spec.action_features)

    contrib_records = _build_records(contribution_rows, contribution_features)
    action_records = _build_records(action_rows, action_features)

    y_contrib = contribution_rows["actual_contribution_bin5"].astype(int).to_numpy()
    y_action = action_rows["observed_action_label"].astype(str).to_numpy()

    contribution_model = _build_classifier(class_weight="balanced")
    action_model = _build_classifier(class_weight="balanced")

    contribution_model.fit(contrib_records, y_contrib)
    action_model.fit(action_records, y_action)

    return {
        "name": family_spec.name,
        "description": family_spec.description,
        "contribution_features": contribution_features,
        "action_features": action_features,
        "contribution_model": contribution_model,
        "action_model": action_model,
        "contribution_classes": [int(value) for value in contribution_model.named_steps["model"].classes_.tolist()],
        "action_classes": [str(value) for value in action_model.named_steps["model"].classes_.tolist()],
        "n_contribution_rows": int(len(contribution_rows)),
        "n_action_rows": int(len(action_rows)),
    }


def train_family_policy_bundle(
    *,
    contribution_stage_path: Path,
    action_stage_path: Path,
    output_model_path: Path = DEFAULT_MODEL_OUTPUT_PATH,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT_PATH,
    metadata_output_path: Path = DEFAULT_METADATA_OUTPUT_PATH,
) -> Dict[str, Any]:
    contribution_df = pd.read_parquet(contribution_stage_path)
    action_df = pd.read_parquet(action_stage_path)

    family_payloads: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for family_spec in FAMILY_LIBRARY:
        family_payload = _fit_family_models(
            contribution_df=contribution_df,
            action_df=action_df,
            family_spec=family_spec,
        )
        family_payloads[family_spec.name] = family_payload
        summary_rows.append(
            {
                "family": family_spec.name,
                "description": family_spec.description,
                "n_contribution_rows": int(family_payload["n_contribution_rows"]),
                "n_action_rows": int(family_payload["n_action_rows"]),
                "n_contribution_features": int(len(family_payload["contribution_features"])),
                "n_action_features": int(len(family_payload["action_features"])),
                "contribution_classes": json.dumps(family_payload["contribution_classes"]),
                "action_classes": json.dumps(family_payload["action_classes"]),
            }
        )

    bundle = {
        "version": 1,
        "source_tables": {
            "contribution_stage": str(contribution_stage_path),
            "action_stage": str(action_stage_path),
        },
        "family_order": [family_spec.name for family_spec in FAMILY_LIBRARY],
        "families": family_payloads,
    }

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, output_model_path)
    pd.DataFrame(summary_rows).to_csv(summary_output_path, index=False)

    metadata = {
        "version": 1,
        "source_tables": {
            "contribution_stage": str(contribution_stage_path),
            "action_stage": str(action_stage_path),
        },
        "n_contribution_rows": int(len(contribution_df)),
        "n_action_rows": int(len(action_df)),
        "families": summary_rows,
    }
    with open(metadata_output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit first-pass algorithm-family contribution/action policies from state tables."
    )
    parser.add_argument(
        "--contribution_stage_path",
        type=str,
        default=str(DEFAULT_STATE_TABLE_ROOT / "learning_wave_contribution_stage.parquet"),
    )
    parser.add_argument(
        "--action_stage_path",
        type=str,
        default=str(DEFAULT_STATE_TABLE_ROOT / "learning_wave_action_stage.parquet"),
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default=str(DEFAULT_MODEL_OUTPUT_PATH),
    )
    parser.add_argument(
        "--summary_output_path",
        type=str,
        default=str(DEFAULT_SUMMARY_OUTPUT_PATH),
    )
    parser.add_argument(
        "--metadata_output_path",
        type=str,
        default=str(DEFAULT_METADATA_OUTPUT_PATH),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metadata = train_family_policy_bundle(
        contribution_stage_path=Path(args.contribution_stage_path).resolve(),
        action_stage_path=Path(args.action_stage_path).resolve(),
        output_model_path=Path(args.output_model_path).resolve(),
        summary_output_path=Path(args.summary_output_path).resolve(),
        metadata_output_path=Path(args.metadata_output_path).resolve(),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
