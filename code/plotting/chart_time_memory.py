"""
Generate Time vs Memory Usage Scatter Plot

Creates scatter plot comparing runtime and memory usage across models.

Creates:
- charts/time_memory.png - Time vs Memory scatter plot

Usage:
    python plotting/chart_time_memory.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings for 2-column paper
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 12

DATASETS = ['agr', 'airlines', 'elec', 'hyp', 'poker', 'rbf', 'rtg', 'sea', 'spam', 'weather']

# Model colors
MODEL_COLORS = {
    'AUTODD': '#d62728',      # Red
    'MetaADD': '#1f77b4',     # Blue
    'GaussianNB': '#7f7f7f',  # Gray
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
    'GaussianNB': 'No Detector (*)',
    'ADWIN': 'ADWIN',
    'DDM': 'DDM',
    'EDDM': 'EDDM',
    'HDDM_A': 'HDDM-A',
    'HDDM_W': 'HDDM-W',
    'PAGEHINKLEY': 'Page-Hinkley',
    'KSWIN': 'KSWIN',
}

# Fixed order: AutoDD first, Meta-ADD second, then others
MODEL_ORDER = ['AUTODD', 'MetaADD', 'GaussianNB', 'ADWIN', 'DDM', 'EDDM', 'HDDM_A', 'HDDM_W', 'PAGEHINKLEY', 'KSWIN']


def load_results():
    """Load all benchmark results."""
    records = []
    for filepath in glob(os.path.join(RESULTS_DIR, "*.json")):
        if 'summary' in os.path.basename(filepath):
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Convert total runtime to ms per instance
            total_runtime = data.get('total_runtime', 0)
            n_instances = data.get('n_instances', 1)
            ms_per_instance = (total_runtime / n_instances) * 1000 if n_instances > 0 else 0
            records.append({
                'model': data.get('model', ''),
                'dataset': data.get('dataset', ''),
                'runtime': ms_per_instance,  # ms per instance
                'memory_mb': data.get('total_memory_mb', 0),
            })
        except:
            continue
    return records


def compute_mean_stats(records):
    """Compute mean time and memory for each model."""
    model_stats = {}

    for rec in records:
        model = rec['model']
        if not model:
            continue
        if model not in model_stats:
            model_stats[model] = {'runtime': [], 'memory': []}
        model_stats[model]['runtime'].append(rec['runtime'])
        model_stats[model]['memory'].append(rec['memory_mb'])

    mean_stats = {}
    for model, stats in model_stats.items():
        if stats['runtime'] and stats['memory']:
            mean_stats[model] = {
                'runtime': np.mean(stats['runtime']),
                'memory': np.mean(stats['memory'])
            }

    return mean_stats


def generate_time_memory_plot():
    """Generate time vs memory scatter plot."""
    records = load_results()
    if not records:
        print("No results found!")
        return None

    mean_stats = compute_mean_stats(records)
    if not mean_stats:
        print("No valid stats computed!")
        return None

    print(f"Models found: {list(mean_stats.keys())}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use fixed order: AutoDD first, Meta-ADD second, then others
    sorted_models = [m for m in MODEL_ORDER if m in mean_stats]
    # Add any models not in MODEL_ORDER at the end
    for m in mean_stats.keys():
        if m not in sorted_models:
            sorted_models.append(m)

    # Plot each model in fixed order
    for model in sorted_models:
        stats = mean_stats[model]
        time_val = stats['runtime']
        memory_val = stats['memory']

        color = MODEL_COLORS.get(model, 'gray')
        name = MODEL_NAMES.get(model, model)

        ax.scatter(
            time_val, memory_val,
            color=color,
            label=name,
            s=150,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5,
            marker='s'
        )

    # Labels
    ax.set_xlabel('Runtime (ms/instance)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=18, fontweight='bold')

    # Legend
    ax.legend(
        loc='upper left',
        framealpha=0.9,
        fontsize=11,
        markerscale=0.8
    )

    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "time_memory.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Time vs Memory Usage Scatter Plot")
    print("=" * 60)

    path = generate_time_memory_plot()
    if path:
        print(f"\nChart saved: {path}")
    else:
        print("Failed to generate chart")

    print("\nDone!")


if __name__ == "__main__":
    main()
