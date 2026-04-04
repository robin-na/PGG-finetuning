#!/usr/bin/env python3
"""Analyze non-LLM trajectory baselines for MobLab task2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_moblab_persistence_and_correlation import (
    MEASURE_LABELS,
    build_all_rounds,
    configure_plot_style,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / 'non-PGG_generalization'
    / 'pgg_transfer_eval'
    / 'output'
    / 'moblab_task2_trajectory_baseline'
)
TASK2_SUPPORTED_MEASURES = ['dictator', 'trust_investor', 'trust_banker', 'pg_contribution']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0:
        return float('nan')
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def spearman_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float('nan')
    left = pd.Series(y_true).rank(method='average')
    right = pd.Series(y_pred).rank(method='average')
    return float(left.corr(right))


def build_long_table() -> pd.DataFrame:
    rounds_df = build_all_rounds()
    rows: List[Dict[str, object]] = []
    for (measure, user_id, session_id), group in rounds_df.groupby(['measure', 'UserID', 'session_id'], sort=False):
        if measure not in TASK2_SUPPORTED_MEASURES:
            continue
        ordered = group.sort_values('Round')
        values = ordered['value'].astype(float).to_numpy()
        rounds = ordered['Round'].astype(int).to_numpy()
        if len(values) < 2:
            continue
        first_value = float(values[0])
        for idx in range(1, len(values)):
            rows.append(
                {
                    'measure': str(measure),
                    'label': MEASURE_LABELS[str(measure)],
                    'UserID': int(user_id),
                    'session_id': str(session_id),
                    'session_rounds': int(len(values)),
                    'future_round': int(rounds[idx]),
                    'horizon_step': int(idx),
                    'actual_share': float(values[idx] * 100.0),
                    'predicted_share_persistence': float(first_value * 100.0),
                }
            )
    table = pd.DataFrame(rows)
    table['abs_error'] = (table['actual_share'] - table['predicted_share_persistence']).abs()
    table['sq_error'] = (table['actual_share'] - table['predicted_share_persistence']) ** 2
    return table


def summarize_by_horizon(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (measure, horizon_step), group in long_df.groupby(['measure', 'horizon_step'], sort=False):
        y = group['actual_share'].to_numpy(dtype=float)
        pred = group['predicted_share_persistence'].to_numpy(dtype=float)
        rows.append(
            {
                'measure': measure,
                'label': MEASURE_LABELS[measure],
                'horizon_step': int(horizon_step),
                'n_values': int(len(group)),
                'n_sessions': int(group['session_id'].nunique()),
                'mean_actual_share': float(np.mean(y)),
                'mean_pred_share': float(np.mean(pred)),
                'mae': float(np.mean(np.abs(y - pred))),
                'rmse': float(np.sqrt(np.mean((y - pred) ** 2))),
                'r2': r2_score_manual(y, pred),
                'spearman': spearman_manual(y, pred),
            }
        )
    return pd.DataFrame(rows).sort_values(['measure', 'horizon_step']).reset_index(drop=True)


def summarize_overall(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for measure, group in long_df.groupby('measure', sort=False):
        y = group['actual_share'].to_numpy(dtype=float)
        pred = group['predicted_share_persistence'].to_numpy(dtype=float)
        per_session = group.groupby('session_id', sort=False)['abs_error'].mean()
        rows.append(
            {
                'measure': measure,
                'label': MEASURE_LABELS[measure],
                'n_values': int(len(group)),
                'n_sessions': int(group['session_id'].nunique()),
                'median_session_rounds': float(group[['session_id', 'session_rounds']].drop_duplicates()['session_rounds'].median()),
                'mae': float(np.mean(np.abs(y - pred))),
                'rmse': float(np.sqrt(np.mean((y - pred) ** 2))),
                'r2': r2_score_manual(y, pred),
                'spearman': spearman_manual(y, pred),
                'sequence_mae': float(per_session.mean()),
            }
        )
    order = {m: i for i, m in enumerate(TASK2_SUPPORTED_MEASURES)}
    out = pd.DataFrame(rows)
    out['order'] = out['measure'].map(order)
    return out.sort_values('order').drop(columns='order').reset_index(drop=True)


def build_summary_md(overall_df: pd.DataFrame, horizon_df: pd.DataFrame) -> str:
    lines = [
        '# MobLab Task2 Trajectory Persistence Baseline',
        '',
        '## Setup',
        '',
        '- Baseline: use the session\'s round-1 action as the prediction for every later round.',
        '- Metrics are computed on share percent (0-100).',
        '- `mae/rmse/r2/spearman` are reported both after flattening all future rounds and by horizon step.',
        '- `sequence_mae` is the average within-session MAE after comparing the full predicted trajectory against the realized future trajectory.',
        '',
        '## Overall Flattened Results',
        '',
        '| Measure | Values | Sessions | Median rounds | MAE | RMSE | R^2 | Spearman | Sequence MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in overall_df.itertuples():
        lines.append(
            f'| {row.label} | {row.n_values} | {row.n_sessions} | {row.median_session_rounds:.1f} | '
            f'{row.mae:.3f} | {row.rmse:.3f} | {row.r2:.3f} | {row.spearman:.3f} | {row.sequence_mae:.3f} |'
        )

    lines.extend(['', '## Horizon-by-Horizon Highlights (Sample-Supported)', ''])
    for measure in TASK2_SUPPORTED_MEASURES:
        sub = horizon_df[(horizon_df['measure'] == measure) & (horizon_df['n_sessions'] >= 100)].copy()
        if sub.empty:
            continue
        best = sub.sort_values('mae').iloc[0]
        worst = sub.sort_values('mae', ascending=False).iloc[0]
        lines.append(
            f'- {MEASURE_LABELS[measure]}: best horizon step {int(best.horizon_step)} has MAE {best.mae:.3f} and R^2 {best.r2:.3f}; '
            f'worst horizon step {int(worst.horizon_step)} has MAE {worst.mae:.3f} and R^2 {worst.r2:.3f}.'
        )
    lines.extend(['', '## First Five Future Steps', ''])
    for measure in TASK2_SUPPORTED_MEASURES:
        sub = horizon_df[horizon_df['measure'] == measure].copy().sort_values('horizon_step').head(5)
        if sub.empty:
            continue
        lines.extend(
            [
                '',
                f'### {MEASURE_LABELS[measure]}',
                '',
                '| Horizon step | Sessions | MAE | R^2 |',
                '|---:|---:|---:|---:|',
            ]
        )
        for row in sub.itertuples():
            lines.append(f'| {row.horizon_step} | {row.n_sessions} | {row.mae:.3f} | {row.r2:.3f} |')
    return '\n'.join(lines) + '\n'


def plot_horizon_curves(horizon_df: pd.DataFrame, out_path: Path) -> None:
    measures = [m for m in TASK2_SUPPORTED_MEASURES if m in set(horizon_df['measure'])]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=False)
    axes = axes.flatten()
    for ax, measure in zip(axes, measures):
        sub = horizon_df[horizon_df['measure'] == measure].copy()
        ax.plot(sub['horizon_step'], sub['mae'], color='#1f6f8b', marker='o', label='MAE')
        ax2 = ax.twinx()
        ax2.plot(sub['horizon_step'], sub['r2'], color='#c44536', marker='s', label='R^2')
        ax.set_title(MEASURE_LABELS[measure])
        ax.set_xlabel('Future horizon step')
        ax.set_ylabel('MAE')
        ax2.set_ylabel('R^2')
        ax.axhline(0.0, color='#8d8d8d', linewidth=0.8, linestyle='--')
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc='upper left', fontsize=9)
    for ax in axes[len(measures):]:
        ax.axis('off')
    fig.suptitle('Task2 Trajectory Persistence Baseline by Horizon', y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    long_df = build_long_table()
    horizon_df = summarize_by_horizon(long_df)
    overall_df = summarize_overall(long_df)

    long_df.to_csv(args.output_dir / 'trajectory_persistence_long.csv', index=False)
    horizon_df.to_csv(args.output_dir / 'trajectory_persistence_by_horizon.csv', index=False)
    overall_df.to_csv(args.output_dir / 'trajectory_persistence_overall.csv', index=False)
    (args.output_dir / 'summary.md').write_text(build_summary_md(overall_df, horizon_df), encoding='utf-8')
    plot_horizon_curves(horizon_df, plots_dir / 'trajectory_persistence_horizon_curves.png')

    print((args.output_dir / 'summary.md').read_text(encoding='utf-8'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
