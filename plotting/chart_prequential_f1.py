"""
Generate Prequential F1 Score Charts

Creates F1 score over time charts with drift points marked.

Creates:
- charts/prequential_f1_<dataset>.png - Per-dataset F1 curves
- charts/prequential_f1_<dataset>_<base_model>.png - Per base model charts

Usage:
    python plotting/chart_prequential_f1.py
    python plotting/chart_prequential_f1.py --base_model hat
    python plotting/chart_prequential_f1.py --base_model all
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings for 2-column paper - bigger, bolder fonts
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 18

DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg', 'elec', 'weather', 'spam', 'airlines', 'poker']
SYNTHETIC = ['sea', 'hyp', 'agr', 'rbf', 'rtg']
BASE_MODELS = ['gaussiannb', 'hat']

# True drift points for synthetic datasets
TRUE_DRIFT_POINTS = {
    'sea': [125000, 250000, 375000],
    'hyp': [150000, 300000],
    'agr': [25000, 50000, 75000],
    'rbf': [66666, 133333],
    'rtg': [20000, 40000, 60000, 80000],
}

# Model display names and colors
MODEL_CONFIG = {
    'AUTODD': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2.5, 'label': 'AutoDD'},
    'MetaADD': {'color': '#1f77b4', 'linestyle': '--', 'linewidth': 2.0, 'label': 'Meta-ADD'},
    'PUDD': {'color': '#ff6b6b', 'linestyle': '-', 'linewidth': 2.0, 'label': 'PUDD'},
    'ADDM': {'color': '#4ecdc4', 'linestyle': '-', 'linewidth': 2.0, 'label': 'ADDM'},
    'MDDM': {'color': '#95e1d3', 'linestyle': '-', 'linewidth': 2.0, 'label': 'MDDM'},
    'GaussianNB': {'color': '#7f7f7f', 'linestyle': ':', 'linewidth': 2.0, 'label': 'No Drift Detector'},
    'HAT': {'color': '#7f7f7f', 'linestyle': ':', 'linewidth': 2.0, 'label': 'No Drift Detector'},
    'ADWIN': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 1.5, 'label': 'ADWIN'},
    'DDM': {'color': '#bcbd22', 'linestyle': '-', 'linewidth': 1.5, 'label': 'DDM'},
    'EDDM': {'color': '#9467bd', 'linestyle': '-', 'linewidth': 1.5, 'label': 'EDDM'},
    'HDDM_A': {'color': '#8c564b', 'linestyle': '-', 'linewidth': 1.5, 'label': 'HDDM-A'},
    'HDDM_W': {'color': '#17becf', 'linestyle': '-', 'linewidth': 1.5, 'label': 'HDDM-W'},
    'PAGEHINKLEY': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 1.5, 'label': 'Page-Hinkley'},
    'KSWIN': {'color': '#e377c2', 'linestyle': '--', 'linewidth': 1.5, 'label': 'KSWIN'},
}

MODEL_ORDER = ['AUTODD', 'MetaADD', 'PUDD', 'ADDM', 'MDDM', 'GaussianNB', 'HAT', 'ADWIN', 'DDM', 'EDDM', 'HDDM_A', 'HDDM_W', 'PAGEHINKLEY', 'KSWIN']


def load_results(base_model=None):
    """Load all benchmark results."""
    results = {}
    for filepath in glob(os.path.join(RESULTS_DIR, "*.json")):
        if 'summary' in os.path.basename(filepath):
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            record_base_model = data.get('base_model', 'gaussiannb')

            if base_model and record_base_model != base_model:
                continue

            model = data.get('model', '')
            dataset = data.get('dataset', '')
            if not model or not dataset:
                continue
            key = (model, dataset)
            if key not in results:
                results[key] = []
            results[key].append(data)
        except Exception as e:
            continue
    return results


def get_mean_f1_curve(runs):
    """Get mean F1 curve from multiple runs."""
    all_instances = []
    all_scores = []

    for run in runs:
        if 'window_metrics' in run:
            wm = run['window_metrics']
            instances = wm.get('instances', [])
            scores = wm.get('f1_scores', [])
            if instances and scores:
                all_instances.append(instances)
                all_scores.append(scores)

    if not all_scores:
        return None, None

    instances = np.array(all_instances[0])
    min_len = min(len(s) for s in all_scores)
    scores_array = np.array([s[:min_len] for s in all_scores])
    mean_scores = np.mean(scores_array, axis=0) * 100

    return instances[:min_len], mean_scores


def plot_scores(dataset, results, base_model=None):
    """Plot prequential F1 scores for a dataset."""

    model_data = {}
    for (model, ds), runs in results.items():
        if ds != dataset:
            continue
        instances, scores = get_mean_f1_curve(runs)
        if instances is not None and scores is not None:
            model_data[model] = {'instances': instances, 'scores': scores}

    if not model_data:
        print(f"No window_metrics data for {dataset}")
        return None

    sorted_models = [m for m in MODEL_ORDER if m in model_data]
    for m in model_data.keys():
        if m not in sorted_models:
            sorted_models.append(m)

    fig, ax = plt.subplots(figsize=(6, 6))
    all_y = []

    for model in sorted_models:
        data = model_data[model]
        instances = data['instances']
        scores = data['scores']
        all_y.extend(scores)

        config = MODEL_CONFIG.get(model, {
            'color': 'gray', 'linestyle': '-', 'linewidth': 1.5, 'label': model
        })

        ax.plot(instances, scores,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                label=config['label'])

    if dataset in TRUE_DRIFT_POINTS:
        for dp in TRUE_DRIFT_POINTS[dataset]:
            ax.axvline(x=dp, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('#Instances', fontsize=18, fontweight='bold')
    ax.set_ylabel('Prequential F1 Score (%)', fontsize=18, fontweight='bold')

    if all_y:
        y_min = min(all_y)
        y_max = max(all_y)
        padding = (y_max - y_min) * 0.05
        ax.set_ylim(max(0, y_min - padding - 2), min(100, y_max + padding + 2))

    ax.legend(fontsize=10, loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()

    if base_model:
        output_path = os.path.join(CHARTS_DIR, f"prequential_f1_{dataset}_{base_model}.png")
    else:
        output_path = os.path.join(CHARTS_DIR, f"prequential_f1_{dataset}.png")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main(base_model='all'):
    print("=" * 60)
    print("Generating Prequential F1 Score Charts")
    print("=" * 60)

    if base_model == 'all':
        base_models_to_run = BASE_MODELS
    else:
        base_models_to_run = [base_model]

    for bm in base_models_to_run:
        print(f"\n--- Processing {bm.upper()} ---")

        results = load_results(base_model=bm)
        print(f"Loaded results for {len(results)} model-dataset combinations")

        for dataset in tqdm.tqdm(DATASETS, desc="Generating charts"):
            path = plot_scores(dataset, results, base_model=bm if len(base_models_to_run) > 1 else None)
            if path:
                print(f"Chart saved: {path}")
            else:
                print(f"Skipped: {dataset} (no data)")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Prequential F1 Score Charts')
    parser.add_argument('--base_model', type=str, default='all',
                        choices=['gaussiannb', 'hat', 'all'],
                        help='Base classifier to use (default: all)')
    args = parser.parse_args()

    main(base_model=args.base_model)
