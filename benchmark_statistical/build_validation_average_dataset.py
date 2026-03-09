from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "benchmark_statistical" / "data" / "processed_data" / "df_analysis_val.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_statistical" / "data" / "processed_data" / "df_analysis_val_averaged.csv"
GROUP_COL = "CONFIG_treatmentName"


def build_averaged_validation_table(input_csv: Path = DEFAULT_INPUT, output_csv: Path = DEFAULT_OUTPUT) -> Path:
    df = pd.read_csv(input_csv)
    if GROUP_COL not in df.columns:
        raise ValueError(f"{input_csv} is missing required column '{GROUP_COL}'.")

    df = df.loc[:, [column for column in df.columns if not str(column).startswith("Unnamed:")]].copy()
    df["__source_order"] = range(len(df))

    group_order = (
        df.groupby(GROUP_COL, as_index=False)["__source_order"]
        .min()
        .sort_values("__source_order")
        [GROUP_COL]
        .tolist()
    )

    numeric_cols = [column for column in df.columns if column != GROUP_COL and pd.api.types.is_numeric_dtype(df[column])]
    object_cols = [column for column in df.columns if column not in numeric_cols and column != "__source_order"]

    first_rows = (
        df.sort_values("__source_order")
        .groupby(GROUP_COL, sort=False, as_index=False)[object_cols]
        .first()
    )
    mean_rows = (
        df.groupby(GROUP_COL, sort=False, as_index=False)[numeric_cols]
        .mean()
        if numeric_cols
        else pd.DataFrame({GROUP_COL: group_order})
    )
    counts = df.groupby(GROUP_COL, sort=False).size().rename("benchmark_game_count").reset_index()

    averaged = first_rows.merge(mean_rows, on=GROUP_COL, how="left").merge(counts, on=GROUP_COL, how="left")
    averaged["__group_order"] = pd.Categorical(averaged[GROUP_COL], categories=group_order, ordered=True)
    averaged = averaged.sort_values("__group_order").drop(columns=["__group_order"]).reset_index(drop=True)

    ordered_columns = [column for column in df.columns if column not in {"__source_order"} and not str(column).startswith("Unnamed:")]
    output_columns = [column for column in ordered_columns if column in averaged.columns]
    if "benchmark_game_count" not in output_columns:
        output_columns.append("benchmark_game_count")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    averaged.loc[:, output_columns].to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    out_path = build_averaged_validation_table()
    print(f"Wrote averaged validation benchmark -> {out_path}")


if __name__ == "__main__":
    main()
