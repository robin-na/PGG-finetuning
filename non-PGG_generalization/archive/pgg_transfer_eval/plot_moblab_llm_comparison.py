#!/usr/bin/env python3
"""Aggregate and plot MobLab LLM-vs-baseline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = PROJECT_ROOT / 'non-PGG_generalization' / 'pgg_transfer_eval' / 'output' / 'moblab_llm' / 'evals'
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_ROOT / 'comparison'
BASELINE_ORDER = ['direct', 'persona', 'meta_persona', 'retrieval']
TASK_ORDER = [('task1', 'scalar'), ('task2', 'future_mean'), ('task2', 'trajectory')]
MEASURE_ORDER = ['dictator', 'pg_contribution', 'trust_banker', 'trust_investor', 'ultimatum_proposer', 'ultimatum_responder']
EXCLUDED_MEASURES = {'pg_contribution'}
MEASURE_LABELS = {
    'dictator': 'Dictator',
    'pg_contribution': 'PGG',
    'trust_banker': 'Trust Banker',
    'trust_investor': 'Trust Investor',
    'ultimatum_proposer': 'Ultimatum Prop.',
    'ultimatum_responder': 'Ultimatum Resp.',
}
COLORS = {
    'direct': '#1f6f8b',
    'persona': '#d17b0f',
    'meta_persona': '#b23a48',
    'retrieval': '#2a9d8f',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_metrics(eval_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(eval_root.glob('*/metrics_by_target.csv')):
        if path.parent.name.startswith('mock_'):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df['source_dir'] = path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[~combined['target_measure'].isin(EXCLUDED_MEASURES)].copy()
    combined['delta_r2_vs_stat'] = combined['r2'] - combined['stat_r2']
    combined['delta_mae_vs_stat'] = combined['mae'] - combined['stat_mae']
    return combined


def task_title(task_type: str, prediction_mode: str) -> str:
    if task_type == 'task1':
        return 'Task1 First-Round Prediction'
    if prediction_mode == 'future_mean':
        return 'Task2 Future Mean'
    return 'Task2 Trajectory'


def grouped_bar(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path) -> None:
    if df.empty:
        return
    measures = [m for m in MEASURE_ORDER if m in set(df['target_measure'])]
    baselines = [b for b in BASELINE_ORDER if b in set(df['baseline'])]
    width = 0.18
    x = list(range(len(measures)))
    fig, ax = plt.subplots(figsize=(1.7 * len(measures) + 3, 5.5))
    for idx, baseline in enumerate(baselines):
        vals = []
        for measure in measures:
            sub = df[(df['baseline'] == baseline) & (df['target_measure'] == measure)]
            vals.append(float(sub.iloc[0][value_col]) if not sub.empty else float('nan'))
        offset = (idx - (len(baselines) - 1) / 2.0) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=baseline, color=COLORS.get(baseline, None))
    ax.set_xticks(x, [MEASURE_LABELS[m] for m in measures], rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(4, len(baselines)))
    ax.axhline(0.0, color='#888', linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    lines = ['# MobLab LLM Comparison', '']
    if df.empty:
        lines.append('- No evaluation files found.')
        out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return
    for task_type, prediction_mode in TASK_ORDER:
        sub = df[(df['task_type'] == task_type) & (df['prediction_mode'] == prediction_mode)].copy()
        if sub.empty:
            continue
        lines.extend(['## ' + task_title(task_type, prediction_mode), '', '| Baseline | Target | MAE | R^2 | Spearman | Baseline MAE | Baseline R^2 | ΔMAE | ΔR^2 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
        sub = sub.sort_values(['target_measure', 'baseline'])
        for row in sub.itertuples():
            lines.append(f'| {row.baseline} | {row.target_measure} | {row.mae:.3f} | {row.r2:.3f} | {row.spearman:.3f} | {row.stat_mae:.3f} | {row.stat_r2:.3f} | {row.delta_mae_vs_stat:.3f} | {row.delta_r2_vs_stat:.3f} |')
        lines.append('')
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_metrics(args.eval_root)
    if not df.empty:
        df.to_csv(args.output_dir / 'all_metrics_combined.csv', index=False)
        for task_type, prediction_mode in TASK_ORDER:
            sub = df[(df['task_type'] == task_type) & (df['prediction_mode'] == prediction_mode)].copy()
            if sub.empty:
                continue
            stem = f'{task_type}_{prediction_mode}'
            grouped_bar(sub, 'r2', f'{task_title(task_type, prediction_mode)}: R^2 by Baseline', 'R^2', args.output_dir / f'{stem}_r2.png')
            grouped_bar(sub, 'mae', f'{task_title(task_type, prediction_mode)}: MAE by Baseline', 'MAE (share percent)', args.output_dir / f'{stem}_mae.png')
            grouped_bar(sub, 'delta_r2_vs_stat', f'{task_title(task_type, prediction_mode)}: ΔR^2 vs Statistical Baseline', 'ΔR^2', args.output_dir / f'{stem}_delta_r2.png')
    write_summary(df, args.output_dir / 'summary.md')
    print(args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
