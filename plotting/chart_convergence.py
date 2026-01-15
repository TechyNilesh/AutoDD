"""
Generate Convergence Analysis Chart

Creates:
- charts/convergence.png - Selection accuracy over stream position

Usage:
    python plotting/chart_convergence.py
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

# Plot settings - publication quality
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11


def load_convergence_data():
    """Load convergence data from JSON."""
    filepath = os.path.join(RESULTS_DIR, 'supplementary_convergence.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    data = load_convergence_data()

    positions = data['positions']
    datasets = list(data['datasets'].keys())

    colors = {
        'SEA': '#1f77b4',
        'HYP': '#ff7f0e',
        'AGR': '#2ca02c',
        'Elec': '#d62728',
        'Poker': '#9467bd',
    }

    markers = {
        'SEA': 'o',
        'HYP': 's',
        'AGR': '^',
        'Elec': 'D',
        'Poker': 'v',
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(positions))

    for dataset in datasets:
        ax.plot(x, data['datasets'][dataset], marker=markers.get(dataset, 'o'),
                color=colors.get(dataset, 'gray'), linewidth=2.5, markersize=10, label=dataset)

    # Mean line
    ax.plot(x, data['mean'], marker='*', color='black', linewidth=3,
            markersize=12, label='Mean', linestyle='--')

    ax.set_xlabel('Stream Position', fontsize=14, fontweight='bold')
    ax.set_ylabel('Selection Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylim(60, 100)

    ax.legend(loc='lower right', framealpha=0.95, fontsize=11)

    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "convergence.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    main()
