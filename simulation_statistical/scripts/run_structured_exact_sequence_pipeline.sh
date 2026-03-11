#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MICRO_RUN_ID="${MICRO_RUN_ID:-260310_val40_dedup_exact_sequence_structured_archetype}"
MACRO_RUN_ID="${MACRO_RUN_ID:-260310_val40_treatment_avg_exact_sequence_structured_archetype}"
MICRO_REPORT_ID="${MICRO_REPORT_ID:-260310_val40_dedup_exact_sequence_structured_vs_history_vs_archetype_vs_random_report}"
MACRO_REPORT_ID="${MACRO_REPORT_ID:-260310_val40_treatment_avg_exact_sequence_structured_vs_history_vs_archetype_vs_random_report}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

echo "[1/5] train structured exact-sequence policy"
python simulation_statistical/train_exact_sequence_policy.py

echo "[2/5] run micro validation simulation"
python simulation_statistical/micro/run_micro_simulation.py \
  --strategy exact_sequence_archetype \
  --run_id "$MICRO_RUN_ID"

echo "[3/5] run macro validation simulation"
python simulation_statistical/macro/run_macro_simulation.py \
  --strategy exact_sequence_archetype \
  --run_id "$MACRO_RUN_ID"

echo "[4/5] build micro comparison report"
python simulation_statistical/micro/analysis/run_analysis.py \
  --compare_run_ids "$MICRO_RUN_ID,260310_val40_dedup_history_archetype,260310_val40_dedup_archetype_cluster,260309_val40_dedup_random_baseline" \
  --compare_labels "exact_sequence_structured,history_archetype,archetype_cluster,random_baseline" \
  --analysis_run_id "$MICRO_REPORT_ID"

echo "[5/5] build macro comparison report"
python simulation_statistical/macro/analysis/run_analysis.py \
  --compare_run_ids "$MACRO_RUN_ID,260310_val40_treatment_avg_history_archetype,260310_val40_treatment_avg_archetype_cluster,260309_val40_treatment_avg_random_baseline" \
  --compare_labels "exact_sequence_structured,history_archetype,archetype_cluster,random_baseline" \
  --analysis_run_id "$MACRO_REPORT_ID" \
  --shared_games_only

echo "done"
echo "micro report: benchmark_statistical/micro/reports/$MICRO_REPORT_ID"
echo "macro report: benchmark_statistical/macro/reports/$MACRO_REPORT_ID"
