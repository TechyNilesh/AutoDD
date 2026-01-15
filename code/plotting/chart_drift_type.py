"""
Generate Per-Drift-Type Performance Chart

Creates:
- charts/drift_type_performance.png - Grouped bar chart

Usage:
    python plotting/chart_drift_type.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 10


def load_drift_type_data():
    """Load drift type performance data from JSON."""
    filepath = os.path.join(RESULTS_DIR, 'supplementary_drift_type.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    data = load_drift_type_data()

    drift_types = data['drift_types']
    methods = data['methods']
    performance = data['performance']

    x = np.arange(len(drift_types))
    width = 0.07
    # Color by category: Classical (blue), SOTA (orange), AutoDD (red)
    colors = ['#7aa6c2'] * 7 + ['#f9a26c'] * 4 + ['#d62728']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        values = [performance[dt][i] for dt in drift_types]
        offset = (i - 5.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i],
                      edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(drift_types, fontweight='bold')
    ax.set_ylim(65, 92)

    ax.legend(loc='upper right', ncol=3, framealpha=0.95, fontsize=9)

    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "drift_type_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    main()
