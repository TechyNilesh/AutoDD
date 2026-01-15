"""
Generate Detection Delay Analysis Chart

Creates:
- charts/detection_delay.png - Bar chart of detection delay

Usage:
    python plotting/chart_detection_delay.py
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
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 10


def load_detection_delay_data():
    """Load detection delay data from JSON."""
    filepath = os.path.join(RESULTS_DIR, 'supplementary_detection_delay.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    data = load_detection_delay_data()

    methods = data['methods']
    delays = data['delays']

    # Color by category: Classical (blue), SOTA (orange), AutoDD (red)
    colors = ['#7aa6c2'] * 7 + ['#f9a26c'] * 4 + ['#d62728']

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(methods, delays, color=colors, edgecolor='black', linewidth=1.2)

    # Highlight AutoDD
    bars[-1].set_color('#d62728')
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)

    ax.set_ylabel('Mean Detection Delay (instances)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 75)

    ax.tick_params(axis='both', which='major', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Rotate x labels
    ax.set_xticklabels(methods, rotation=45, ha='right', fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Add value labels on bars
    for i, (method, delay) in enumerate(zip(methods, delays)):
        ax.text(i, delay + 1.5, f'{delay:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "detection_delay.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chart saved: {output_path}")


if __name__ == "__main__":
    main()
