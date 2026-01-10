"""
Experiment: Impact of Meta-Feature Types on Detector Selection

Tests AutoDD's detector selection accuracy with different meta-feature configurations:
- Statistical only (5 features)
- Stationarity only (5 features)
- Temporal only (5 features)
- All (15 features)

Uses optimal window size w=100.
Note: Window size sensitivity is tested in run_sensitivity_experiment.py

Usage:
    python run_meta_feature_experiment.py
    python run_meta_feature_experiment.py --n_runs 5
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from river import naive_bayes, metrics, drift

from AutoDD.meta_feature_extractor import MetaFeatureExtractor
from AutoDD.detector_portfolio import DetectorPortfolio
from AutoDD.linucb_agent import LinUCBAgent

import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "experiment_results"

# Dataset configuration
DATASET_INFO = {
    'path': 'datasets/synthetic_streams/',
    'n_streams': 100,
    'n_train': 70,
    'n_test': 30,
    'drift_types': {
        'sudden': 25,      # streams 001-025
        'gradual': 25,     # streams 026-050
        'incremental': 25, # streams 051-075
        'recurring': 25    # streams 076-100
    },
    'instances_per_stream': 10000
}

# Feature indices
STATISTICAL_INDICES = [0, 1, 2, 3, 4]
STATIONARITY_INDICES = [5, 6, 7, 8, 9]
TEMPORAL_INDICES = [10, 11, 12, 13, 14]
ALL_INDICES = list(range(15))

# Configurations to test
FEATURE_CONFIGS = {
    'statistical': ('Statistical only', STATISTICAL_INDICES),
    'stationarity': ('Stationarity only', STATIONARITY_INDICES),
    'temporal': ('Temporal only', TEMPORAL_INDICES),
    'all': ('All (15 features)', ALL_INDICES),
}

# Use optimal window size (sensitivity tested separately)
WINDOW_SIZE = 100


def get_drift_type_from_stream_id(stream_id):
    """Determine drift type based on stream ID."""
    if stream_id <= 25:
        return 'sudden'
    elif stream_id <= 50:
        return 'gradual'
    elif stream_id <= 75:
        return 'incremental'
    else:
        return 'recurring'


def load_stream_from_file(filepath):
    """Load stream data from file."""
    data = []
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('#'):
            parts = first_line.split(',')
            x = {f'f{i}': float(v) for i, v in enumerate(parts[:-1])}
            y = int(float(parts[-1]))
            data.append((x, y))

        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                x = {f'f{i}': float(v) for i, v in enumerate(parts[:-1])}
                y = int(float(parts[-1]))
                data.append((x, y))
    return data


def load_synthetic_streams(data_dir, n_streams=100):
    """Load synthetic streams from directory."""
    streams = []
    stream_files = sorted(glob(os.path.join(data_dir, 'stream_*.csv')))

    if not stream_files:
        stream_files = sorted(glob(os.path.join(data_dir, '*.csv')))

    if not stream_files:
        print(f"Warning: No stream files found in {data_dir}")
        return None

    for i, filepath in enumerate(stream_files[:n_streams]):
        stream_id = i + 1
        drift_type = get_drift_type_from_stream_id(stream_id)
        streams.append({
            'filepath': filepath,
            'stream_id': stream_id,
            'drift_type': drift_type
        })

    return streams


def generate_synthetic_stream(drift_type, n_instances=10000, seed=42):
    """Generate a synthetic stream with specified drift pattern (fallback)."""
    from river.datasets import synth
    np.random.seed(seed)

    if drift_type == 'sudden':
        stream = synth.Agrawal(classification_function=0, seed=seed)
    elif drift_type == 'gradual':
        stream = synth.Hyperplane(seed=seed, n_features=10, n_drift_features=5, mag_change=0.001)
    elif drift_type == 'incremental':
        stream = synth.SEA(variant=0, seed=seed, noise=0.1)
    else:
        stream = synth.Mixed(classification_function=0, seed=seed, balance_classes=True)

    return stream


class MetaFeatureAutoDD:
    """AutoDD with configurable meta-features for ablation study."""

    def __init__(self, window_size=100, alpha=1.0, feature_indices=None):
        self.window_size = window_size
        self.feature_indices = feature_indices if feature_indices else ALL_INDICES

        self.meta_extractor = MetaFeatureExtractor(window_size=window_size)
        self.portfolio = DetectorPortfolio()

        self.n_arms = len(self.portfolio)
        self.n_features = len(self.feature_indices)

        self.agent = LinUCBAgent(
            n_arms=self.n_arms,
            n_features=self.n_features,
            alpha=alpha
        )

        self.arm_to_detector = {i: name for i, name in enumerate(self.portfolio.detector_names)}
        self.selection_history = []
        self._n_instances = 0

    def _select_features(self, full_features):
        return full_features[self.feature_indices]

    def get_selected_detector(self, x, y_true, y_pred):
        """Update and return selected detector name."""
        self._n_instances += 1
        self.meta_extractor.update(x, y_true, y_pred)

        if self.meta_extractor.is_ready():
            full_features = self.meta_extractor.get_features()
            features = self._select_features(full_features)
            arm = self.agent.select_arm(features)
            detector_name = self.arm_to_detector[arm]
            self.selection_history.append(detector_name)
            return detector_name

        return self.portfolio.detector_names[0]


def run_meta_feature_experiment(data_dir=None, n_runs=1, seed=42):
    """Run meta-feature type impact experiment."""

    if data_dir is None:
        data_dir = DATASET_INFO['path']

    np.random.seed(seed)

    print("=" * 70)
    print("Experiment: Impact of Meta-Feature Types")
    print("=" * 70)
    print(f"Dataset: {data_dir}")
    print(f"Streams: {DATASET_INFO['n_streams']} ({DATASET_INFO['n_train']} train, {DATASET_INFO['n_test']} test)")
    print(f"Window size: {WINDOW_SIZE} (optimal)")
    print(f"Runs: {n_runs}")
    print()

    # Load streams
    streams = load_synthetic_streams(data_dir, DATASET_INFO['n_streams'])
    use_files = streams is not None

    if not use_files:
        print("Generating synthetic streams...")
        streams = []
        drift_types = ['sudden', 'gradual', 'incremental', 'recurring']
        for i in range(DATASET_INFO['n_streams']):
            drift_type = drift_types[i % len(drift_types)]
            streams.append({
                'stream_id': i + 1,
                'drift_type': drift_type,
                'seed': seed + i
            })

    results = defaultdict(list)

    for run in range(n_runs):
        run_seed = seed + run * 100
        np.random.seed(run_seed)

        print(f"\n--- Run {run + 1}/{n_runs} ---")

        train_streams = streams[:DATASET_INFO['n_train']]
        test_streams = streams[DATASET_INFO['n_train']:]

        for config_name, (display_name, feature_indices) in FEATURE_CONFIGS.items():

            correct_selections = 0
            total_selections = 0

            for stream_info in test_streams:
                if use_files and 'filepath' in stream_info:
                    data = load_stream_from_file(stream_info['filepath'])
                else:
                    stream = generate_synthetic_stream(
                        stream_info['drift_type'],
                        DATASET_INFO['instances_per_stream'],
                        seed=stream_info.get('seed', run_seed + stream_info['stream_id'])
                    )
                    data = list(stream.take(DATASET_INFO['instances_per_stream']))

                autodd = MetaFeatureAutoDD(
                    window_size=WINDOW_SIZE,
                    feature_indices=feature_indices
                )

                model = naive_bayes.GaussianNB()

                for i, (x, y) in enumerate(data):
                    if i >= DATASET_INFO['instances_per_stream']:
                        break

                    y_pred = model.predict_one(x)
                    selected = autodd.get_selected_detector(x, y, y_pred)

                    if i > 0 and i % WINDOW_SIZE == 0:
                        total_selections += 1
                        if selected in ['ADWIN', 'KSWIN', 'DDM']:
                            correct_selections += 1

                    model.learn_one(x, y)

            if total_selections > 0:
                accuracy = correct_selections / total_selections
            else:
                accuracy = 0.0

            results[config_name].append(accuracy)
            print(f"  {display_name:20s} | Acc: {accuracy:.2%}")

    # Aggregate results
    final_results = {}

    print("\n" + "=" * 70)
    print("SUMMARY: Selection Accuracy by Meta-Feature Type")
    print("=" * 70)
    print(f"\n{'Meta-Features':<25s} {'Accuracy':>12s} {'Std':>10s}")
    print("-" * 50)

    for config_name, (display_name, _) in FEATURE_CONFIGS.items():
        mean_acc = np.mean(results[config_name])
        std_acc = np.std(results[config_name]) if n_runs > 1 else 0
        final_results[config_name] = {
            'mean': round(mean_acc, 4),
            'std': round(std_acc, 4)
        }
        print(f"{display_name:<25s} {mean_acc*100:>10.1f}% {std_acc*100:>9.1f}%")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "meta_feature_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'Meta-Feature Types Impact',
            'dataset_info': DATASET_INFO,
            'window_size': WINDOW_SIZE,
            'n_runs': n_runs,
            'results': final_results
        }, f, indent=4)

    print(f"\nResults saved to: {output_file}")

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meta-Feature Types Impact Experiment')
    parser.add_argument('--data_dir', type=str, default=None,
                        help=f"Dataset directory (default: {DATASET_INFO['path']})")
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    run_meta_feature_experiment(
        data_dir=args.data_dir,
        n_runs=args.n_runs,
        seed=args.seed
    )
