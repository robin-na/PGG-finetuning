from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def cast_bool_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and out[col].dtype == bool:
            out[col] = out[col].astype(int)
    return out


FIXED_CONFIG_FEATURES = [
    "CONFIG_playerCount",
    "CONFIG_numRounds",
    "CONFIG_showNRounds",
    "CONFIG_allOrNothing",
    "CONFIG_chat",
    "CONFIG_defaultContribProp",
    "CONFIG_punishmentCost",
    "CONFIG_punishmentTech",
    "CONFIG_rewardExists",
    "CONFIG_rewardCost",
    "CONFIG_rewardTech",
    "CONFIG_showOtherSummaries",
    "CONFIG_showPunishmentId",
    "CONFIG_showRewardId",
    "CONFIG_MPCR",
]

RIDGE_ALPHA = 1.0


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    paired_learn_path = root / "benchmark_sequential" / "data" / "processed_data" / "df_paired_learn.csv"
    paired_val_path = root / "benchmark_sequential" / "data" / "processed_data" / "df_paired_val.csv"
    seed_summary_path = (
        root
        / "benchmark_sequential"
        / "data_seed_random_learning_paired300_pairs10_rs42"
        / "seed_selection_summary.json"
    )

    out_dir = root / "benchmark_sequential" / "results" / "results_seed_vs_full_ridge_rs42"
    out_dir.mkdir(parents=True, exist_ok=True)

    learn = pd.read_csv(paired_learn_path)
    val = pd.read_csv(paired_val_path)
    seed_summary = json.loads(seed_summary_path.read_text(encoding="utf-8"))

    seed_pair_bases = {str(x) for x in seed_summary["seed_pair_bases"]}
    seed_learn = learn[learn["CONFIG_configId"].astype(str).isin(seed_pair_bases)].copy()

    if len(seed_learn) != 10:
        raise ValueError(f"Expected 10 seed pairs in paired-learn table, found {len(seed_learn)}")

    # User-specified fixed feature set for all future runs.
    config_features = FIXED_CONFIG_FEATURES.copy()
    features = config_features + ["control_itt_efficiency"]
    target = "treatment_itt_efficiency"

    for required_col in features + [target]:
        if required_col not in learn.columns or required_col not in val.columns:
            raise ValueError(f"Missing required column `{required_col}` in learn/val paired data.")

    learn = cast_bool_features(learn, features)
    seed_learn = cast_bool_features(seed_learn, features)
    val = cast_bool_features(val, features)

    model_full = Ridge(alpha=RIDGE_ALPHA, random_state=0)
    model_seed = Ridge(alpha=RIDGE_ALPHA, random_state=0)
    model_full.fit(learn[features], learn[target])
    model_seed.fit(seed_learn[features], seed_learn[target])

    val_pred_full = model_full.predict(val[features])
    val_pred_seed = model_seed.predict(val[features])
    val_pred_base = val["control_itt_efficiency"].to_numpy()

    mse_full = float(mean_squared_error(val[target], val_pred_full))
    mse_seed = float(mean_squared_error(val[target], val_pred_seed))
    mse_base_zero_treatment_effect = float(mean_squared_error(val[target], val_pred_base))
    rmse_full = float(np.sqrt(mse_full))
    rmse_seed = float(np.sqrt(mse_seed))
    r2_full_custom = float(1 - (mse_full / mse_base_zero_treatment_effect))
    r2_seed_custom = float(1 - (mse_seed / mse_base_zero_treatment_effect))

    predictions = val.copy()
    predictions["pred_treatment_itt_efficiency_full150"] = val_pred_full
    predictions["pred_treatment_itt_efficiency_seed10"] = val_pred_seed
    predictions["pred_treatment_itt_efficiency_base_zero_treatment_effect"] = val_pred_base
    predictions["abs_err_full150"] = (predictions[target] - predictions["pred_treatment_itt_efficiency_full150"]).abs()
    predictions["abs_err_seed10"] = (predictions[target] - predictions["pred_treatment_itt_efficiency_seed10"]).abs()
    predictions["abs_err_base_zero_treatment_effect"] = (
        predictions[target] - predictions["pred_treatment_itt_efficiency_base_zero_treatment_effect"]
    ).abs()
    predictions.to_csv(out_dir / "val_predictions_full_vs_seed.csv", index=False)

    summary = {
        "paired_learn_path": str(paired_learn_path),
        "paired_val_path": str(paired_val_path),
        "seed_summary_path": str(seed_summary_path),
        "seed_random_state": seed_summary.get("random_state"),
        "regression_model": "ridge",
        "ridge_alpha": RIDGE_ALPHA,
        "selected_config_features": config_features,
        "n_selected_config_features": len(config_features),
        "features_used": features,
        "rows": {
            "learn_pairs_full": int(len(learn)),
            "learn_pairs_seed": int(len(seed_learn)),
            "val_pairs": int(len(val)),
        },
        "rmse": {
            "full150_on_val": rmse_full,
            "seed10_on_val": rmse_seed,
            "seed_minus_full": rmse_seed - rmse_full,
        },
        "mse": {
            "base_zero_treatment_effect_on_val": mse_base_zero_treatment_effect,
            "full150_on_val": mse_full,
            "seed10_on_val": mse_seed,
        },
        "r2_custom": {
            "definition": "1 - MSE_pred / MSE_base, where baseline predicts zero treatment effect (treatment_itt_efficiency = control_itt_efficiency)",
            "full150_on_val": r2_full_custom,
            "seed10_on_val": r2_seed_custom,
        },
    }
    (out_dir / "rmse_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
