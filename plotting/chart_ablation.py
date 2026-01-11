"""
Generate Ablation Study Box Plots

Creates box plots showing ablation study results for F1 and Accuracy.

Creates:
- charts/ablation_f1.png - F1 score box plot
- charts/ablation_accuracy.png - Accuracy box plot

Usage:
    python plotting/chart_ablation.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "ablation_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings for 2-column paper
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14

# Configuration order and display names
CONFIGS = ['full', 'no_temporal', 'no_statistical', 'no_stationarity', 'no_linucb', 'no_thompson']
DISPLAY_NAMES = {
    'full': 'Full AutoDD',
    'no_temporal': 'w/o Temporal',
    'no_statistical': 'w/o Statistical',
    'no_stationarity': 'w/o Stationarity',
    'no_linucb': 'w/o LinUCB\n(Level 1)',
    'no_thompson': 'w/o Thompson\n(Level 2)',
}

# Box colors
BOX_COLORS = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#8c564b']


def load_all_results():
    """Load all ablation results from individual files."""
    results = {config: {'accuracy': [], 'f1': []} for config in CONFIGS}

    for filepath in glob(os.path.join(RESULTS_DIR, "ablation_*.json")):
        if 'summary' in os.path.basename(filepath):
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            config = data.get('config', '')
            if config in CONFIGS:
                results[config]['accuracy'].append(data.get('accuracy', 0) * 100)
                results[config]['f1'].append(data.get('f1', 0) * 100)
        except:
            continue

    return results


def generate_boxplot(results, metric, ylabel, output_name):
    """Generate box plot for a specific metric."""
    # Prepare data
    data = [results[config][metric] for config in CONFIGS]
    labels = [DISPLAY_NAMES[config] for config in CONFIGS]

    # Calculate means and deltas
    means = [np.mean(d) for d in data]
    full_mean = means[0]  # Full AutoDD is baseline
    deltas = [m - full_mean for m in means]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create box plot
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)

    # Color the boxes
    for patch, color in zip(bp['boxes'], BOX_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    # Style whiskers, caps, medians
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='black', linewidth=2.5)
    for flier in bp['fliers']:
        flier.set(marker='o', markerfacecolor='red', markersize=6, alpha=0.6)

    # Add delta annotations above each box
    for i, (d, delta) in enumerate(zip(data, deltas)):
        max_val = np.max(d)
        if i == 0:
            # Full AutoDD - show "Baseline"
            ax.annotate('Baseline',
                       xy=(i + 1, max_val),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', va='bottom', fontsize=11, fontweight='bold',
                       color='green')
        else:
            # Show delta
            color = 'red' if delta < 0 else 'green'
            ax.annotate(f'{delta:+.1f}%',
                       xy=(i + 1, max_val),
                       xytext=(0, 8), textcoords='offset points',
                       ha='center', va='bottom', fontsize=12, fontweight='bold',
                       color=color)

    # Labels
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=12, fontweight='bold')

    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add some padding at top for annotations
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.1)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Ablation Study Box Plots")
    print("=" * 60)

    results = load_all_results()

    # Print data counts
    for config in CONFIGS:
        print(f"{config}: {len(results[config]['accuracy'])} runs")

    # Generate F1 box plot
    path_f1 = generate_boxplot(results, 'f1', 'F1 Score (%)', 'ablation_f1.png')
    print(f"\nChart saved: {path_f1}")

    # Generate Accuracy box plot
    path_acc = generate_boxplot(results, 'accuracy', 'Accuracy (%)', 'ablation_accuracy.png')
    print(f"Chart saved: {path_acc}")

    print("\nDone!")


if __name__ == "__main__":
    main()
