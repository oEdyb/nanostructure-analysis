import matplotlib.pyplot as plt
import apd_functions
import numpy as np
from scipy.signal import savgol_filter
import re

import spectra_functions
from spectra_functions import compute_sensitivity_stat


def plot_sensitivity_points_by_group(
    group_value_store,
    output_path,
    group_stats=None,
    xtick_formatter=None,
    secondary_group_values=None,
    secondary_ylabel='SNR'
):
    """Violin distribution per group for sensitivity (and optional secondary metric)."""
    if not group_value_store:
        return

    groups = sorted(group_value_store)
    datasets = [('Sensitivity (mV/nm)', group_value_store)]
    if secondary_group_values:
        datasets.append((secondary_ylabel, secondary_group_values))

    stats = group_stats or {}
    colors = ['#2C7BB6', '#ABD9E9', '#FDAE61', '#D7191C']
    color_map = {grp: colors[i % len(colors)] for i, grp in enumerate(groups)}

    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 5), squeeze=False)
    axes = axes.ravel()

    for ax, (ylabel, value_store) in zip(axes, datasets):
        values_per_group = [np.asarray(value_store.get(group, []), dtype=float) for group in groups]
        violins = ax.violinplot(
            values_per_group,
            positions=np.arange(len(groups)),
            widths=0.6,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        for body, group in zip(violins['bodies'], groups):
            color = color_map[group]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)

        for idx, (group, values) in enumerate(zip(groups, values_per_group)):
            if not values.size:
                continue
            median = float(np.median(values))
            spread = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            ax.fill_between([idx - 0.16, idx + 0.16], median - spread, median + spread, color=color_map[group], alpha=0.14, zorder=1)
            ax.hlines([median - spread, median + spread], idx - 0.12, idx + 0.12, colors='#555555', linestyles='--', linewidth=1.0, alpha=0.9, zorder=2)
            ax.scatter(idx, median, s=55, color='#111111', marker='D', edgecolors='white', linewidths=0.7, zorder=3)

        labels = [
            (
                f"{group}\nGap {stats[group]['gap_mean']:.1f}±{stats[group].get('gap_std', 0.0):.1f} nm\n"
                f"Tip {stats[group]['tip_mean']:.3f}±{stats[group].get('tip_std', 0.0):.3f}"
            ) if group in stats else (xtick_formatter(group) if xtick_formatter else group)
            for group in groups
        ]
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', color='#E4E4E4', linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_vs_sem_metrics_group(
    group_values,
    group_stats,
    output_path,
    xlabel_gap='Gap Mean (nm)',
    xlabel_tip='Tip Curvature Mean',
    ylabel='Sensitivity (mV/nm)',
    secondary_group_values=None,
    secondary_ylabel='SNR'
):
    """Scatter sensitivity per group vs. gap/tip metrics with error bars."""
    if not group_values or not group_stats:
        return

    groups = sorted(set(group_values) & set(group_stats))
    if not groups:
        return

    datasets = [(group_values, ylabel)]
    if secondary_group_values:
        datasets.append((secondary_group_values, secondary_ylabel))

    colors = ['#2C7BB6', '#ABD9E9', '#FDAE61', '#D7191C']
    color_map = {grp: colors[i % len(colors)] for i, grp in enumerate(groups)}

    gap_mean = np.asarray([group_stats[g]['gap_mean'] for g in groups], dtype=float)
    gap_std = np.asarray([group_stats[g].get('gap_std', 0.0) for g in groups], dtype=float)
    tip_mean = np.asarray([group_stats[g]['tip_mean'] for g in groups], dtype=float)
    tip_std = np.asarray([group_stats[g].get('tip_std', 0.0) for g in groups], dtype=float)

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 4.8 * len(datasets)), squeeze=False)

    for row, (values_dict, row_label) in enumerate(datasets):
        values_per_group = [np.asarray(values_dict.get(group, []), dtype=float) for group in groups]
        medians = np.asarray([float(np.median(vals)) if vals.size else np.nan for vals in values_per_group], dtype=float)
        spreads = np.asarray([float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0 for vals in values_per_group], dtype=float)

        for col, (x_vals, x_err, xlabel) in enumerate(((gap_mean, gap_std, xlabel_gap), (tip_mean, tip_std, xlabel_tip))):
            ax = axes[row, col]
            for idx, group in enumerate(groups):
                y_val = medians[idx]
                if np.isnan(y_val):
                    continue
                color = color_map[group]
                ax.errorbar(
                    x_vals[idx],
                    y_val,
                    yerr=spreads[idx],
                    xerr=x_err[idx],
                    fmt='o',
                    color=color,
                    ecolor=color,
                    capsize=4,
                    markersize=8
                )
                ax.annotate(group, (x_vals[idx], y_val), textcoords="offset points", xytext=(4, 6), fontsize=9)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.grid(color='#E4E4E4', linewidth=0.7, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[row, 0].set_ylabel(row_label, fontsize=12, fontweight='bold')

    fig.suptitle('Metrics vs. Group Statistics', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_vs_sem_metrics_points(
    measurement_series,
    sem_measurements,
    output_path,
    metrics=None,
    sensitivity_fn=None,
    ylabel='Sensitivity P95 |mV/nm|',
    secondary_measurement_values=None,
    secondary_ylabel='SNR 3x3'
):
    """Scatter per-measurement sensitivity vs. SEM metrics with optional regression."""
    if not measurement_series or not sem_measurements:
        return

    shared_ids = [mid for mid in sorted(measurement_series) if mid in sem_measurements]
    if not shared_ids:
        return

    metrics = metrics or [('gap_mean', 'Gap Mean (nm)'), ('tip_mean', 'Tip Curvature Mean')]
    sensitivity_fn = sensitivity_fn or compute_sensitivity_stat

    color_cycle = ['#2C7BB6', '#ABD9E9', '#FDAE61', '#D7191C']
    group_colors = {}
    entries = []
    for measurement_id in shared_ids:
        values = np.asarray(measurement_series[measurement_id].get('values', []), dtype=float)
        if not values.size:
            continue
        group = measurement_series[measurement_id].get('group') or 'Unknown'
        entry = {
            'id': measurement_id,
            'group': group,
            'sensitivity': float(sensitivity_fn(values)),
            'sem': sem_measurements[measurement_id]
        }
        if secondary_measurement_values and measurement_id in secondary_measurement_values:
            entry['secondary'] = float(secondary_measurement_values[measurement_id])
        entries.append(entry)
        if group not in group_colors:
            group_colors[group] = color_cycle[len(group_colors) % len(color_cycle)]

    if not entries:
        return

    rows = 2 if any('secondary' in entry for entry in entries) else 1
    fig, axes = plt.subplots(rows, len(metrics), figsize=(5.5 * len(metrics), 4.8 * rows), squeeze=False)

    def add_correlation(ax, x_vals, y_vals):
        if len(x_vals) < 2:
            return
        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2 or np.unique(x).size < 2:
            return
        coeff = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.02, 0.95, f'r = {coeff:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        try:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            ax.plot(grid, slope * grid + intercept, color='#444444', linewidth=1.1, alpha=0.6)
        except np.linalg.LinAlgError:
            pass

    def render_row(row_idx, value_key, row_label):
        for col, metric_cfg in enumerate(metrics):
            metric_key, metric_label = metric_cfg[:2]
            show_corr = bool(metric_cfg[2]) if len(metric_cfg) > 2 else False
            ax = axes[row_idx, col]
            x_vals, y_vals = [], []
            for entry in entries:
                if value_key not in entry:
                    continue
                x_val = entry['sem'].get(metric_key)
                y_val = entry[value_key]
                if x_val is None or y_val is None:
                    continue
                color = group_colors.get(entry['group'], '#333333')
                ax.scatter(x_val, y_val, color=color, s=50)
                ax.annotate(entry['id'], (x_val, y_val), textcoords="offset points", xytext=(4, 5), fontsize=8)
                x_vals.append(x_val)
                y_vals.append(y_val)
            if show_corr:
                add_correlation(ax, x_vals, y_vals)
            ax.set_xlabel(metric_label, fontsize=12)
            ax.grid(color='#E4E4E4', linewidth=0.7, alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, fontweight='bold')

    render_row(0, 'sensitivity', ylabel)
    if rows == 2:
        render_row(1, 'secondary', secondary_ylabel)

    fig.suptitle('Per-Measurement Metrics vs. SEM Features', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
