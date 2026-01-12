"""
Generate Lambda (Regularization) Sensitivity Chart

Creates:
- charts/lambda_sensitivity.png - Line chart showing accuracy vs lambda values

Usage:
    python plotting/chart_lambda_sensitivity.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(CODE_DIR, "experiment_results", "lambda_sensitivity_summary.json")
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

DATASET_NAMES = {
    'sea': 'SEA',
    'hyp': 'HYP',
    'agr': 'AGR',
    'rbf': 'RBF',
    'rtg': 'RTG',
    'elec': 'ELEC',
}

DATASET_COLORS = {
    'sea': '#1f77b4',    # Blue
    'hyp': '#ff7f0e',    # Orange
    'agr': '#2ca02c',    # Green
    'rbf': '#d62728',    # Red
    'rtg': '#9467bd',    # Purple
    'elec': '#8c564b',   # Brown
}

DATASET_MARKERS = {
    'sea': 'o',
    'hyp': 's',
    'agr': '^',
    'rbf': 'D',
    'rtg': 'v',
    'elec': 'p',
}


def load_results():
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def plot_lambda_sensitivity(data):
    results = data['results']
    lambda_values = data['lambda_values']
    optimal = data.get('optimal_lambda', 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = list(results.keys())
    for ds in datasets:
        accuracies = []
        stds = []
        for lam in lambda_values:
            lam_key = str(lam)
            accuracies.append(results[ds][lam_key]['accuracy_mean'] * 100)
            stds.append(results[ds][lam_key]['accuracy_std'] * 100)

        color = DATASET_COLORS.get(ds, 'gray')
        marker = DATASET_MARKERS.get(ds, 'o')
        name = DATASET_NAMES.get(ds, ds.upper())

        ax.plot(lambda_values, accuracies, marker=marker, color=color,
                linewidth=3, markersize=12, label=name)
        ax.fill_between(lambda_values,
                        np.array(accuracies) - np.array(stds),
                        np.array(accuracies) + np.array(stds),
                        color=color, alpha=0.15)

    # Mark optimal lambda
    ax.axvline(x=optimal, color='black', linestyle='--', linewidth=2,
               label=f'Optimal (λ={optimal})')

    ax.set_xlabel('Lambda (λ)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=22, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlim(0.008, 6)
    ax.set_ylim(77, 93)

    ax.legend(loc='lower right', framealpha=0.9, ncol=2, fontsize=14)

    ax.tick_params(axis='both', which='major', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, "lambda_sensitivity.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Lambda Sensitivity Chart")
    print("=" * 60)

    data = load_results()
    path = plot_lambda_sensitivity(data)

    print(f"\nChart saved: {path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
