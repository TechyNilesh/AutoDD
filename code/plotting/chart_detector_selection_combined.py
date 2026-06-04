"""
Generate Combined Detector Selection + Error Rate Chart

Creates a single unified visualization showing detector selection
as colored background with error rate overlay.

Creates:
- charts/detector_selection_<dataset>.png - Combined detector selection timeline

Usage:
    python plotting/chart_detector_selection_combined.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings
rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9

# Detector colors - distinct and visible
DETECTOR_COLORS = {
    'ADWIN': '#3498db',      # Blue
    'KSWIN': '#2ecc71',      # Green
    'DDM': '#e74c3c',        # Red
    'PageHinkley': '#9b59b6', # Purple
    'EDDM': '#f39c12',       # Orange
    'HDDM_A': '#1abc9c',     # Teal
    'HDDM_W': '#e91e63',     # Pink
}

DETECTOR_NAMES = {
    'ADWIN': 'ADWIN',
    'KSWIN': 'KSWIN',
    'DDM': 'DDM',
    'PageHinkley': 'Page-Hinkley',
    'EDDM': 'EDDM',
    'HDDM_A': 'HDDM-A',
    'HDDM_W': 'HDDM-W',
}

SYNTHETIC_DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg']


def load_timeline_data(dataset):
    """Load timeline data from JSON."""
    filepath = os.path.join(RESULTS_DIR, f'timeline_{dataset}.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_legend(dataset, filename=None, ncol=None):
    """Standalone single-row legend for the detector-selection timeline, placed
    above the figure in LaTeX (matches the prequential/efficiency shared-legend style)."""
    data = load_timeline_data(dataset)
    used = set(seg['detector'] for seg in data['timeline'])
    detectors_used = [d for d in DETECTOR_COLORS if d in used]
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Error Rate'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Drift Point'),
    ]
    for det in detectors_used:
        legend_elements.append(Patch(facecolor=DETECTOR_COLORS.get(det, 'gray'),
                                     alpha=0.3, edgecolor='gray',
                                     label=DETECTOR_NAMES.get(det, det)))
    n = len(legend_elements)
    if ncol is None:
        ncol = n  # single row
    if filename is None:
        filename = f'detector_selection_legend_{dataset}.png'
    fig = plt.figure(figsize=(2.0 * n, 0.5))
    fig.legend(handles=legend_elements, loc='center', ncol=ncol, frameon=False,
               fontsize=12, handlelength=1.6, handletextpad=0.5, columnspacing=1.3)
    output_path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_combined_chart(dataset, show_legend=True):
    """Plot combined detector selection + error rate chart for a dataset."""
    data = load_timeline_data(dataset)

    n_instances = data['n_instances']
    drift_points = data['drift_points']
    timeline = data['timeline']
    error_x = data['error_rate']['x']
    error_y = data['error_rate']['y']

    # Get unique detectors for legend
    detectors_used = list(set([seg['detector'] for seg in timeline]))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot detector selection as colored background bands
    for seg in timeline:
        start = seg['start']
        end = seg['end']
        detector = seg['detector']
        color = DETECTOR_COLORS.get(detector, 'gray')
        ax.axvspan(start, end, color=color, alpha=0.3, zorder=1)

    # Plot error rate line on top
    ax.plot(error_x, error_y, color='black', linewidth=2, label='Error Rate', zorder=3)

    # Mark drift points with vertical dashed lines (highlighted)
    for i, drift_point in enumerate(drift_points):
        label = 'Drift Point' if i == 0 else None
        ax.axvline(x=drift_point, color='red', linestyle='--', linewidth=2,
                   alpha=0.8, zorder=4, label=label)
        # Add triangle marker at top
        ax.plot(drift_point, max(error_y) * 1.05, 'v', color='red',
                markersize=8, zorder=5)

    # Labels
    ax.set_xlabel('Instance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')

    # Axis limits
    ax.set_xlim(0, n_instances)
    ax.set_ylim(0, max(error_y) * 1.15)

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Error Rate'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Drift Point'),
    ]
    # Add detector patches
    for det in detectors_used:
        color = DETECTOR_COLORS.get(det, 'gray')
        name = DETECTOR_NAMES.get(det, det)
        legend_elements.append(Patch(facecolor=color, alpha=0.3, edgecolor='gray', label=name))

    if show_legend:
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95,
                  ncol=2, fontsize=9)

    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Light grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, f"detector_selection_{dataset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Detector Selection Charts for All Synthetic Datasets")
    print("=" * 60)

    # Generate combined charts for all synthetic datasets
    for dataset in SYNTHETIC_DATASETS:
        try:
            path = plot_combined_chart(dataset, show_legend=False)
            legend = generate_legend(dataset)
            print(f"Chart saved: {path}")
            print(f"Legend saved: {legend}")
        except FileNotFoundError:
            print(f"Skipped {dataset}: timeline data not found")

    print("\nDone!")


if __name__ == "__main__":
    main()
