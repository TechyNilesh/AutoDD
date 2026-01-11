"""
Generate Time vs Memory Usage Trade-off Plot

Creates an enhanced visualization showing runtime vs memory trade-off
with Pareto frontier and efficiency annotations.

Creates:
- charts/time_memory.png - Enhanced time vs memory trade-off plot
- charts/time_memory_<base_model>.png - Per base model plots

Usage:
    python plotting/chart_time_memory.py
    python plotting/chart_time_memory.py --base_model hat
    python plotting/chart_time_memory.py --base_model all
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings for 2-column paper
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 10

DATASETS = ['agr', 'airlines', 'elec', 'hyp', 'poker', 'rbf', 'rtg', 'sea', 'spam', 'weather']
BASE_MODELS = ['gaussiannb', 'hat']
BASE_MODEL_NAMES = {'gaussiannb': 'GaussianNB', 'hat': 'HAT'}

# Model colors - emphasize AutoDD
MODEL_COLORS = {
    'AUTODD': '#d62728',      # Red (highlighted)
    'MetaADD': '#1f77b4',     # Blue
    'PUDD': '#ff6b6b',        # Coral red
    'ADDM': '#4ecdc4',        # Turquoise
    'MDDM': '#95e1d3',        # Mint green
    'GaussianNB': '#7f7f7f',  # Gray
    'HAT': '#7f7f7f',         # Gray
    'ADWIN': '#2ca02c',       # Green
    'DDM': '#bcbd22',         # Yellow-green
    'EDDM': '#9467bd',        # Purple
    'HDDM_A': '#8c564b',      # Brown
    'HDDM_W': '#17becf',      # Cyan
    'PAGEHINKLEY': '#ff7f0e', # Orange
    'KSWIN': '#e377c2',       # Pink
}

# Model display names
MODEL_NAMES = {
    'AUTODD': 'AutoDD',
    'MetaADD': 'Meta-ADD',
    'PUDD': 'PUDD',
    'ADDM': 'ADDM',
    'MDDM': 'MDDM',
    'GaussianNB': 'No Drift Detector',
    'HAT': 'No Drift Detector',
    'ADWIN': 'ADWIN',
    'DDM': 'DDM',
    'EDDM': 'EDDM',
    'HDDM_A': 'HDDM-A',
    'HDDM_W': 'HDDM-W',
    'PAGEHINKLEY': 'Page-Hinkley',
    'KSWIN': 'KSWIN',
}

# Model markers
MODEL_MARKERS = {
    'AUTODD': '*',
    'MetaADD': 'D',
    'PUDD': 'h',
    'ADDM': 'H',
    'MDDM': 'd',
    'GaussianNB': 'X',
    'HAT': 'X',
    'ADWIN': 'o',
    'DDM': 's',
    'EDDM': '^',
    'HDDM_A': 'v',
    'HDDM_W': '<',
    'PAGEHINKLEY': '>',
    'KSWIN': 'p',
}

MODEL_ORDER = ['AUTODD', 'MetaADD', 'PUDD', 'ADDM', 'MDDM', 'ADWIN', 'DDM', 'EDDM', 'HDDM_A', 'HDDM_W', 'PAGEHINKLEY', 'KSWIN']
EXCLUDE_MODELS = ['GaussianNB', 'HAT']


def load_results(base_model=None):
    """Load all benchmark results."""
    records = []
    for filepath in glob(os.path.join(RESULTS_DIR, "*.json")):
        if 'summary' in os.path.basename(filepath):
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            record_base_model = data.get('base_model', 'gaussiannb')
            # Normalize base model name to lowercase
            if record_base_model == 'GaussianNB':
                record_base_model = 'gaussiannb'
            elif record_base_model == 'HAT':
                record_base_model = 'hat'

            if base_model and record_base_model != base_model:
                continue

            total_runtime = data.get('total_runtime', 0)
            n_instances = data.get('n_instances', 1)
            ms_per_instance = (total_runtime / n_instances) * 1000 if n_instances > 0 else 0
            records.append({
                'model': data.get('model', ''),
                'dataset': data.get('dataset', ''),
                'runtime': ms_per_instance,
                'memory_mb': data.get('total_memory_mb', 0),
                'accuracy': data.get('accuracy', 0),
                'base_model': record_base_model,
            })
        except:
            continue
    return records


def compute_mean_stats(records):
    """Compute mean time, memory, and accuracy for each model."""
    model_stats = {}

    for rec in records:
        model = rec['model']
        if not model:
            continue
        if model in EXCLUDE_MODELS:
            continue
        if model not in model_stats:
            model_stats[model] = {'runtime': [], 'memory': [], 'accuracy': []}
        model_stats[model]['runtime'].append(rec['runtime'])
        model_stats[model]['memory'].append(rec['memory_mb'])
        model_stats[model]['accuracy'].append(rec['accuracy'])

    mean_stats = {}
    for model, stats in model_stats.items():
        if stats['runtime'] and stats['memory']:
            mean_stats[model] = {
                'runtime': np.mean(stats['runtime']),
                'memory': np.mean(stats['memory']),
                'accuracy': np.mean(stats['accuracy']) if stats['accuracy'] else 0
            }

    return mean_stats


def generate_time_memory_plot(base_model=None):
    """Generate clean time vs memory trade-off plot with quadrant shading."""
    records = load_results(base_model=base_model)
    if not records:
        print(f"No results found for {base_model}!")
        return None

    mean_stats = compute_mean_stats(records)
    if not mean_stats:
        print("No valid stats computed!")
        return None

    print(f"Models found: {list(mean_stats.keys())}")

    for model, stats in mean_stats.items():
        print(f"  {model}: {stats['runtime']:.3f} ms, {stats['memory']:.1f} MB")

    fig, ax = plt.subplots(figsize=(8, 6))

    sorted_models = [m for m in MODEL_ORDER if m in mean_stats]
    for m in mean_stats.keys():
        if m not in sorted_models:
            sorted_models.append(m)

    time_values = [mean_stats[m]['runtime'] for m in sorted_models]
    memory_values = [mean_stats[m]['memory'] for m in sorted_models]

    # Auto-scale based on data for each base model
    x_max = max(time_values) * 1.15
    y_max = max(memory_values) * 1.1

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    x_mid = x_max / 2
    y_mid = y_max / 2

    ax.fill_between([0, x_mid], [0, 0], [y_mid, y_mid],
                    color='green', alpha=0.12, zorder=0)
    ax.fill_between([x_mid, x_max], [0, 0], [y_mid, y_mid],
                    color='orange', alpha=0.12, zorder=0)
    ax.fill_between([0, x_mid], [y_mid, y_mid], [y_max, y_max],
                    color='orange', alpha=0.12, zorder=0)
    ax.fill_between([x_mid, x_max], [y_mid, y_mid], [y_max, y_max],
                    color='red', alpha=0.12, zorder=0)

    pad = 0.02
    ax.text(x_max * pad, y_mid - y_max * pad, 'Fast + Low Memory', ha='left', va='top',
            fontsize=9, color='darkgreen', alpha=0.7, style='italic', zorder=1)
    ax.text(x_mid + x_max * pad, y_mid - y_max * pad, 'Slow + Low Memory', ha='left', va='top',
            fontsize=9, color='darkorange', alpha=0.7, style='italic', zorder=1)
    ax.text(x_max * pad, y_max - y_max * pad, 'Fast + High Memory', ha='left', va='top',
            fontsize=9, color='darkorange', alpha=0.7, style='italic', zorder=1)
    ax.text(x_mid + x_max * pad, y_max - y_max * pad, 'Slow + High Memory', ha='left', va='top',
            fontsize=9, color='darkred', alpha=0.7, style='italic', zorder=1)

    for model in sorted_models:
        stats = mean_stats[model]
        time_val = stats['runtime']
        memory_val = stats['memory']

        color = MODEL_COLORS.get(model, 'gray')
        name = MODEL_NAMES.get(model, model)
        marker = MODEL_MARKERS.get(model, 'o')

        if model == 'AUTODD':
            size = 400
            edge_width = 2.5
            zorder = 10
        elif model == 'MetaADD':
            size = 300
            edge_width = 2
            zorder = 9
        else:
            size = 200
            edge_width = 1.5
            zorder = 5

        ax.scatter(
            time_val, memory_val,
            color=color,
            label=name,
            s=size,
            alpha=0.9,
            edgecolor='black',
            linewidth=edge_width,
            marker=marker,
            zorder=zorder
        )

    ax.set_xlabel('Runtime (ms/instance)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=13, fontweight='bold')

    legend = ax.legend(
        loc='lower right',
        framealpha=0.95,
        fontsize=9,
        markerscale=0.6,
        ncol=1,
        handletextpad=0.5,
        labelspacing=0.8,
        borderpad=0.8
    )
    legend.get_frame().set_edgecolor('gray')

    ax.tick_params(axis='both', which='major', labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.25, linestyle='--', zorder=1)

    plt.tight_layout()

    if base_model:
        output_path = os.path.join(CHARTS_DIR, f"time_memory_{base_model}.png")
    else:
        output_path = os.path.join(CHARTS_DIR, "time_memory.png")

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def main(base_model='all'):
    print("=" * 60)
    print("Generating Time vs Memory Trade-off Plot")
    print("=" * 60)

    if base_model == 'all':
        base_models_to_run = BASE_MODELS
    else:
        base_models_to_run = [base_model]

    for bm in base_models_to_run:
        print(f"\n--- Processing {bm.upper()} ---")

        path = generate_time_memory_plot(base_model=bm)
        if path:
            print(f"Chart saved: {path}")
        else:
            print(f"Failed to generate chart for {bm}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Time vs Memory Trade-off Plot')
    parser.add_argument('--base_model', type=str, default='all',
                        choices=['gaussiannb', 'hat', 'all'],
                        help='Base classifier to use (default: all)')
    args = parser.parse_args()

    main(base_model=args.base_model)
