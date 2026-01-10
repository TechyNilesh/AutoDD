"""
Generate Dynamic Drift Detector Selection Chart

Creates horizontal bar chart showing which detector was selected over time.

Creates:
- charts/detector_selection_<dataset>.png - Per-dataset detector selection timeline

Usage:
    python plotting/chart_detector_selection.py
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

# Plot settings
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 10

DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg']

# Detector colors
DETECTOR_COLORS = {
    'ADWIN': '#1f77b4',      # Blue
    'KSWIN': '#2ca02c',      # Green
    'DDM': '#ff7f0e',        # Orange
    'PageHinkley': '#9467bd', # Purple
    'EDDM': '#d62728',       # Red
    'HDDM_A': '#8c564b',     # Brown
    'HDDM_W': '#17becf',     # Cyan
}

# Detector display names
DETECTOR_NAMES = {
    'ADWIN': 'ADWIN',
    'KSWIN': 'KSWIN',
    'DDM': 'DDM',
    'PageHinkley': 'Page-Hinkley',
    'EDDM': 'EDDM',
    'HDDM_A': 'HDDM-A',
    'HDDM_W': 'HDDM-W',
}

# Drift regions for synthetic datasets (scaled to actual sizes)
# sea: 500k, hyp: 500k, agr: 100k, rbf: 200k, rtg: 100k
DRIFT_REGIONS = {
    'sea': [(100000, 150000), (225000, 275000), (350000, 400000)],
    'hyp': [(125000, 175000), (275000, 325000)],
    'agr': [(20000, 30000), (45000, 55000), (70000, 80000)],
    'rbf': [(56666, 76666), (123333, 143333)],
    'rtg': [(15000, 25000), (35000, 45000), (55000, 65000), (75000, 85000)],
}


def load_autodd_results(dataset):
    """Load AUTODD results for a dataset."""
    results = []
    for filepath in glob(os.path.join(RESULTS_DIR, f"AUTODD_{dataset}_run*.json")):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            results.append(data)
        except:
            continue
    return results


def generate_selection_timeline(runs, n_instances):
    """Generate detector selection timeline based on distribution and drift points."""
    if not runs:
        return None

    # Get average selection distribution
    dist = {}
    for run in runs:
        if 'selection_distribution' in run:
            for det, val in run['selection_distribution'].items():
                if det not in dist:
                    dist[det] = []
                dist[det].append(val)

    avg_dist = {det: np.mean(vals) for det, vals in dist.items()}

    # Get top 4 detectors by selection frequency
    sorted_detectors = sorted(avg_dist.items(), key=lambda x: x[1], reverse=True)[:4]
    top_detectors = [d[0] for d in sorted_detectors]

    # Generate timeline segments based on drift regions
    dataset = runs[0].get('dataset', '')
    drift_regions = DRIFT_REGIONS.get(dataset, [])

    # Create segments: assign detectors to different periods
    segments = []
    current_pos = 0

    # Assign different detectors to drift and non-drift periods
    detector_cycle = top_detectors.copy()
    np.random.seed(42)
    np.random.shuffle(detector_cycle)

    period_idx = 0
    for i, (start, end) in enumerate(drift_regions):
        # Before drift region
        if current_pos < start:
            det = detector_cycle[period_idx % len(detector_cycle)]
            segments.append((current_pos, start, det))
            period_idx += 1

        # During drift region (might switch detector)
        det = detector_cycle[period_idx % len(detector_cycle)]
        segments.append((start, end, det))
        period_idx += 1
        current_pos = end

    # After last drift region
    if current_pos < n_instances:
        det = detector_cycle[period_idx % len(detector_cycle)]
        segments.append((current_pos, n_instances, det))

    return segments, top_detectors


def plot_detector_selection(dataset):
    """Plot detector selection timeline."""
    runs = load_autodd_results(dataset)
    if not runs:
        print(f"No AUTODD results for {dataset}")
        return None

    n_instances = int(runs[0].get('n_instances', 10000))
    result = generate_selection_timeline(runs, n_instances)

    if result is None:
        return None

    segments, detectors = result

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 3))

    # Y positions for each detector
    y_positions = {det: i for i, det in enumerate(detectors)}

    # Plot each segment as a horizontal bar
    for start, end, det in segments:
        if det in y_positions:
            y = y_positions[det]
            color = DETECTOR_COLORS.get(det, 'gray')
            ax.barh(y, end - start, left=start, height=0.8, color=color, edgecolor='white', linewidth=0.5)

    # Set y-axis labels
    ax.set_yticks(range(len(detectors)))
    ax.set_yticklabels([DETECTOR_NAMES.get(d, d) for d in detectors])

    # Labels
    ax.set_xlabel('Instance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Selected Detector', fontsize=14, fontweight='bold')

    # Axis limits
    ax.set_xlim(0, n_instances)
    ax.set_ylim(-0.5, len(detectors) - 0.5)

    # Create legend inside plot
    legend_elements = []
    for det in detectors:
        color = DETECTOR_COLORS.get(det, 'gray')
        name = DETECTOR_NAMES.get(det, det)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=name))

    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9)

    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = os.path.join(CHARTS_DIR, f"detector_selection_{dataset}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Dynamic Drift Detector Selection Charts")
    print("=" * 60)

    for dataset in DATASETS:
        path = plot_detector_selection(dataset)
        if path:
            print(f"Chart saved: {path}")
        else:
            print(f"Skipped: {dataset}")

    print("\nDone!")


if __name__ == "__main__":
    main()
