"""
Generate Window Size Sensitivity Chart

Creates:
- charts/window_sensitivity.png - Line chart showing accuracy vs window sizes

Usage:
    python plotting/chart_window_sensitivity.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(CODE_DIR, "experiment_results", "window_size_analysis_summary.json")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings - larger fonts for subfigure display
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 22
rcParams['axes.labelsize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 16

CONFIG_NAMES = {
    'statistical': 'Statistical',
    'stationarity': 'Stationarity',
    'temporal': 'Temporal',
    'all': 'All Features',
}

CONFIG_COLORS = {
    'statistical': '#1f77b4',    # Blue
    'stationarity': '#ff7f0e',   # Orange
    'temporal': '#2ca02c',       # Green
    'all': '#d62728',            # Red
}

CONFIG_MARKERS = {
    'statistical': 'o',
    'stationarity': 's',
    'temporal': '^',
    'all': 'D',
}


def load_results():
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def plot_window_sensitivity(data):
    results = data['results']
    window_sizes = data['window_sizes']
    optimal = data.get('optimal_window', 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ['statistical', 'stationarity', 'temporal', 'all']
    for config in configs:
        if config not in results:
            continue

        accuracies = []
        stds = []
        for ws in window_sizes:
            ws_key = str(ws)
            accuracies.append(results[config][ws_key]['mean'] * 100)
            stds.append(results[config][ws_key]['std'] * 100)

        color = CONFIG_COLORS.get(config, 'gray')
        marker = CONFIG_MARKERS.get(config, 'o')
        name = CONFIG_NAMES.get(config, config)

        ax.plot(window_sizes, accuracies, marker=marker, color=color,
                linewidth=3, markersize=12, label=name)
        ax.fill_between(window_sizes,
                        np.array(accuracies) - np.array(stds),
                        np.array(accuracies) + np.array(stds),
                        color=color, alpha=0.15)

    # Mark optimal window
    ax.axvline(x=optimal, color='black', linestyle='--', linewidth=1.5,
               label=f'Optimal (w={optimal})')

    ax.set_xlabel('Window Size (w)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Selection Accuracy (%)', fontsize=22, fontweight='bold')

    ax.set_xlim(20, 210)
    ax.set_ylim(58, 98)

    ax.legend(loc='lower right', framealpha=0.9, fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "window_sensitivity.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Window Size Sensitivity Chart")
    print("=" * 60)

    data = load_results()
    path = plot_window_sensitivity(data)

    print(f"\nChart saved: {path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
