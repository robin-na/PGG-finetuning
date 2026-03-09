from __future__ import annotations

import os


BENCHMARK_STATISTICAL_ROOT = "benchmark_statistical"
BENCHMARK_DATA_ROOT = os.path.join(BENCHMARK_STATISTICAL_ROOT, "data")

MICRO_RUN_ROOT = os.path.join(BENCHMARK_STATISTICAL_ROOT, "micro", "runs")
MICRO_REPORT_ROOT = os.path.join(BENCHMARK_STATISTICAL_ROOT, "micro", "reports")

MACRO_RUN_ROOT = os.path.join(BENCHMARK_STATISTICAL_ROOT, "macro", "runs")
MACRO_REPORT_ROOT = os.path.join(BENCHMARK_STATISTICAL_ROOT, "macro", "reports")


def analysis_csv_name_for_wave(wave: str) -> str:
    return "df_analysis_val_averaged.csv" if wave == "validation_wave" else "df_analysis_learn.csv"


def demographics_csv_name_for_wave(wave: str) -> str:
    return "demographics_numeric_val.csv" if wave == "validation_wave" else "demographics_numeric_learn.csv"

