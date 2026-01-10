"""
Generate Time vs Memory Usage Trade-off Plot

Creates an enhanced visualization showing runtime vs memory trade-off
with Pareto frontier and efficiency annotations.

Creates:
- charts/time_memory.png - Enhanced time vs memory trade-off plot

Usage:
    python plotting/chart_time_memory.py
"""

import os
import json
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

# Model colors - emphasize AutoDD
MODEL_COLORS = {
    'AUTODD': '#d62728',      # Red (highlighted)
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

# Model markers - different shapes for different method types
MODEL_MARKERS = {
    'AUTODD': '*',        # Star for our method
    'MetaADD': 'D',       # Diamond for meta-learning
    'GaussianNB': 'X',    # X for baseline
    'ADWIN': 'o',         # Circle for detectors
    'DDM': 's',           # Square
    'EDDM': '^',          # Triangle up
    'HDDM_A': 'v',        # Triangle down
    'HDDM_W': '<',        # Triangle left
    'PAGEHINKLEY': '>',   # Triangle right
    'KSWIN': 'p',         # Pentagon
}

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
            total_runtime = data.get('total_runtime', 0)
            n_instances = data.get('n_instances', 1)
            ms_per_instance = (total_runtime / n_instances) * 1000 if n_instances > 0 else 0
            records.append({
                'model': data.get('model', ''),
                'dataset': data.get('dataset', ''),
                'runtime': ms_per_instance,
                'memory_mb': data.get('total_memory_mb', 0),
                'accuracy': data.get('accuracy', 0),
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


def find_pareto_frontier(points):
    """Find Pareto-optimal points (minimize both time and memory)."""
    pareto_points = []
    for i, (x1, y1, name1) in enumerate(points):
        dominated = False
        for j, (x2, y2, name2) in enumerate(points):
            if i != j and x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):
                dominated = True
                break
        if not dominated:
            pareto_points.append((x1, y1, name1))
    # Sort by x for line drawing
    pareto_points.sort(key=lambda p: p[0])
    return pareto_points


def generate_time_memory_plot():
    """Generate clean time vs memory trade-off plot with quadrant shading."""
    records = load_results()
    if not records:
        print("No results found!")
        return None

    mean_stats = compute_mean_stats(records)
    if not mean_stats:
        print("No valid stats computed!")
        return None

    print(f"Models found: {list(mean_stats.keys())}")

    # Print stats for debugging
    for model, stats in mean_stats.items():
        print(f"  {model}: {stats['runtime']:.3f} ms, {stats['memory']:.1f} MB")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use fixed order
    sorted_models = [m for m in MODEL_ORDER if m in mean_stats]
    for m in mean_stats.keys():
        if m not in sorted_models:
            sorted_models.append(m)

    # Get all values for axis limits
    time_values = [mean_stats[m]['runtime'] for m in sorted_models]
    memory_values = [mean_stats[m]['memory'] for m in sorted_models]

    # Set axis limits with padding
    x_max = max(time_values) * 1.15
    y_max = max(memory_values) * 1.1
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Calculate midpoints for quadrants
    x_mid = x_max / 2
    y_mid = y_max / 2

    # Draw 4 transparent quadrant boxes (zorder=0 to be behind everything)
    # Bottom-left: Fast + Low Memory (Green - Best)
    ax.fill_between([0, x_mid], [0, 0], [y_mid, y_mid],
                    color='green', alpha=0.12, zorder=0)
    # Bottom-right: Slow + Low Memory (Yellow)
    ax.fill_between([x_mid, x_max], [0, 0], [y_mid, y_mid],
                    color='orange', alpha=0.12, zorder=0)
    # Top-left: Fast + High Memory (Yellow)
    ax.fill_between([0, x_mid], [y_mid, y_mid], [y_max, y_max],
                    color='orange', alpha=0.12, zorder=0)
    # Top-right: Slow + High Memory (Red - Worst)
    ax.fill_between([x_mid, x_max], [y_mid, y_mid], [y_max, y_max],
                    color='red', alpha=0.12, zorder=0)

    # Add quadrant labels (top-left corner of each box)
    pad = 0.02  # padding from edge
    ax.text(x_max * pad, y_mid - y_max * pad, 'Fast + Low Memory', ha='left', va='top',
            fontsize=9, color='darkgreen', alpha=0.7, style='italic', zorder=1)
    ax.text(x_mid + x_max * pad, y_mid - y_max * pad, 'Slow + Low Memory', ha='left', va='top',
            fontsize=9, color='darkorange', alpha=0.7, style='italic', zorder=1)
    ax.text(x_max * pad, y_max - y_max * pad, 'Fast + High Memory', ha='left', va='top',
            fontsize=9, color='darkorange', alpha=0.7, style='italic', zorder=1)
    ax.text(x_mid + x_max * pad, y_max - y_max * pad, 'Slow + High Memory', ha='left', va='top',
            fontsize=9, color='darkred', alpha=0.7, style='italic', zorder=1)

    # Plot each model - NO text labels, just markers
    for model in sorted_models:
        stats = mean_stats[model]
        time_val = stats['runtime']
        memory_val = stats['memory']

        color = MODEL_COLORS.get(model, 'gray')
        name = MODEL_NAMES.get(model, model)
        marker = MODEL_MARKERS.get(model, 'o')

        # Larger marker for AutoDD and Meta-ADD
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

    # Labels
    ax.set_xlabel('Runtime (ms/instance)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=13, fontweight='bold')

    # Legend inside the chart (lower right - labels moved to top-left of each box)
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

    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Light grid
    ax.grid(True, alpha=0.25, linestyle='--', zorder=1)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "time_memory.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Time vs Memory Trade-off Plot")
    print("=" * 60)

    path = generate_time_memory_plot()
    if path:
        print(f"\nChart saved: {path}")
    else:
        print("Failed to generate chart")

    print("\nDone!")


if __name__ == "__main__":
    main()
