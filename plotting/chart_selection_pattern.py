"""
Generate Detector Selection Patterns Chart

Creates:
- charts/selection_pattern.png - Grouped bar chart

Usage:
    python plotting/chart_selection_pattern.py
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
rcParams['legend.fontsize'] = 11


def load_selection_pattern_data():
    """Load selection pattern data from JSON."""
    filepath = os.path.join(RESULTS_DIR, 'supplementary_selection_pattern.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    data = load_selection_pattern_data()

    detectors = data['detectors']
    conditions = data['conditions']
    frequencies = data['frequencies']

    x = np.arange(len(detectors))
    width = 0.25

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, condition in enumerate(conditions):
        offset = (i - 1) * width
        ax.bar(x + offset, frequencies[condition], width, label=condition,
               color=colors[i], edgecolor='black', linewidth=1)

    ax.set_ylabel('Selection Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, fontweight='bold')
    ax.set_ylim(0, 35)

    ax.legend(loc='upper right', framealpha=0.95, fontsize=11)

    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "selection_pattern.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    main()
