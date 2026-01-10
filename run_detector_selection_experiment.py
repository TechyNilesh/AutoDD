"""
Experiment 2: Detector Selection Patterns Across Drift Types

Investigates whether AutoDD learns interpretable mappings between
stream characteristics and detectors.

Uses synthetic streams from datasets/synthetic_streams/ directory.

Usage:
    python run_detector_selection_experiment.py
    python run_detector_selection_experiment.py --window_size 100
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from river import naive_bayes, metrics

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

# Drift types for analysis (normal added for stationary periods within streams)
DRIFT_TYPES = ['sudden', 'gradual', 'incremental', 'normal']

# All features
ALL_INDICES = list(range(15))

# Default window size (optimal from Experiment 1)
DEFAULT_WINDOW_SIZE = 100


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
    """
    Load stream data from file.
    Returns list of (x, y) tuples.
    """
    data = []
    with open(filepath, 'r') as f:
        # Skip header if exists
        first_line = f.readline().strip()
        if not first_line.startswith('#'):
            # First line is data
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
    """
    Load synthetic streams from directory.
    Returns list of stream info dictionaries.
    """
    streams = []

    # Try to find stream files
    stream_files = sorted(glob(os.path.join(data_dir, 'stream_*.csv')))

    if not stream_files:
        # Try alternate naming
        stream_files = sorted(glob(os.path.join(data_dir, '*.csv')))

    if not stream_files:
        print(f"Warning: No stream files found in {data_dir}")
        print("Falling back to synthetic stream generation...")
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


def generate_labeled_stream(drift_type, n_instances=10000, seed=42):
    """
    Generate a synthetic stream with labeled drift regions (fallback).
    Returns stream and list of (start, end, drift_type) tuples.
    """
    from river.datasets import synth
    np.random.seed(seed)

    drift_regions = []

    if drift_type == 'sudden':
        stream = synth.Agrawal(classification_function=0, seed=seed)
        # Sudden drift at midpoint
        drift_point = n_instances // 2
        drift_regions.append((drift_point - 100, drift_point + 100, 'sudden'))
        # Normal before and after
        drift_regions.append((0, drift_point - 100, 'normal'))
        drift_regions.append((drift_point + 100, n_instances, 'normal'))

    elif drift_type == 'gradual':
        stream = synth.Hyperplane(seed=seed, n_features=10, n_drift_features=5, mag_change=0.001)
        # Gradual drift in middle third
        start = n_instances // 3
        end = 2 * n_instances // 3
        drift_regions.append((start, end, 'gradual'))
        drift_regions.append((0, start, 'normal'))
        drift_regions.append((end, n_instances, 'normal'))

    elif drift_type == 'incremental':
        stream = synth.SEA(variant=0, seed=seed, noise=0.1)
        # Incremental drift throughout
        drift_regions.append((n_instances // 4, 3 * n_instances // 4, 'incremental'))
        drift_regions.append((0, n_instances // 4, 'normal'))
        drift_regions.append((3 * n_instances // 4, n_instances, 'normal'))

    elif drift_type == 'recurring':
        stream = synth.Mixed(classification_function=0, seed=seed, balance_classes=True)
        # Recurring treated as incremental for selection purposes
        drift_regions.append((n_instances // 4, 3 * n_instances // 4, 'incremental'))
        drift_regions.append((0, n_instances // 4, 'normal'))
        drift_regions.append((3 * n_instances // 4, n_instances, 'normal'))

    else:  # normal/stationary
        stream = synth.SEA(variant=0, seed=seed, noise=0.05)
        drift_regions.append((0, n_instances, 'normal'))

    return stream, drift_regions


def get_drift_type_at_position(position, drift_regions):
    """Get the drift type at a specific position."""
    for start, end, dtype in drift_regions:
        if start <= position < end:
            return dtype
    return 'normal'


class SelectionTracker:
    """Tracks detector selections by drift type."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.meta_extractor = MetaFeatureExtractor(window_size=window_size)
        self.portfolio = DetectorPortfolio()

        self.n_arms = len(self.portfolio)
        self.n_features = 15

        self.agent = LinUCBAgent(
            n_arms=self.n_arms,
            n_features=self.n_features,
            alpha=1.0
        )

        self.arm_to_detector = {i: name for i, name in enumerate(self.portfolio.detector_names)}

        # Selection tracking by drift type
        self.selections = {dt: defaultdict(int) for dt in DRIFT_TYPES}
        self.total_by_type = {dt: 0 for dt in DRIFT_TYPES}

        self._n_instances = 0

    def update(self, x, y_true, y_pred, drift_type):
        """Update and track selection."""
        self._n_instances += 1
        self.meta_extractor.update(x, y_true, y_pred)

        if self._n_instances % self.window_size == 0 and self.meta_extractor.is_ready():
            features = self.meta_extractor.get_features()
            arm = self.agent.select_arm(features)
            detector_name = self.arm_to_detector[arm]

            self.selections[drift_type][detector_name] += 1
            self.total_by_type[drift_type] += 1

    def get_selection_frequencies(self):
        """Get selection frequency percentages by drift type."""
        frequencies = {}
        for dtype in DRIFT_TYPES:
            frequencies[dtype] = {}
            total = self.total_by_type[dtype]
            if total > 0:
                for detector in self.portfolio.detector_names:
                    count = self.selections[dtype][detector]
                    frequencies[dtype][detector] = (count / total) * 100
            else:
                for detector in self.portfolio.detector_names:
                    frequencies[dtype][detector] = 0.0
        return frequencies


def run_experiment_2(data_dir=None, window_size=100, seed=42):
    """Run Experiment 2: Detector selection patterns by drift type."""

    if data_dir is None:
        data_dir = DATASET_INFO['path']

    np.random.seed(seed)

    print("=" * 80)
    print("Experiment 2: Detector Selection Patterns Across Drift Types")
    print("=" * 80)
    print(f"Dataset: {data_dir}")
    print(f"Streams: {DATASET_INFO['n_streams']}")
    print(f"Instances per stream: {DATASET_INFO['instances_per_stream']}")
    print(f"Window size: {window_size}")
    print()

    tracker = SelectionTracker(window_size=window_size)

    # Load streams from file or generate
    streams = load_synthetic_streams(data_dir, DATASET_INFO['n_streams'])
    use_files = streams is not None

    if not use_files:
        # Generate streams (fallback)
        print("Generating synthetic streams...")
        streams = []
        stream_drift_types = ['sudden', 'gradual', 'incremental', 'recurring']
        for i in range(DATASET_INFO['n_streams']):
            dtype = stream_drift_types[i % len(stream_drift_types)]
            streams.append({
                'stream_id': i + 1,
                'drift_type': dtype,
                'seed': seed + i
            })

    # Process streams
    for stream_info in streams:
        dtype = stream_info['drift_type']
        print(f"Processing stream {stream_info['stream_id']:3d} ({dtype})...")

        # Load or generate stream data
        if use_files and 'filepath' in stream_info:
            data = load_stream_from_file(stream_info['filepath'])
            # Create simple drift regions for file-based streams
            n_instances = len(data)
            drift_regions = [
                (0, n_instances // 3, 'normal'),
                (n_instances // 3, 2 * n_instances // 3, dtype if dtype != 'recurring' else 'incremental'),
                (2 * n_instances // 3, n_instances, 'normal')
            ]
        else:
            stream, drift_regions = generate_labeled_stream(
                dtype, DATASET_INFO['instances_per_stream'],
                seed=stream_info.get('seed', seed + stream_info['stream_id'])
            )
            data = list(stream.take(DATASET_INFO['instances_per_stream']))

        model = naive_bayes.GaussianNB()

        for pos, (x, y) in enumerate(data):
            if pos >= DATASET_INFO['instances_per_stream']:
                break

            y_pred = model.predict_one(x)
            current_dtype = get_drift_type_at_position(pos, drift_regions)

            tracker.update(x, y, y_pred, current_dtype)

            model.learn_one(x, y)

    # Get and display results
    frequencies = tracker.get_selection_frequencies()

    print("\n" + "=" * 80)
    print("DETECTOR SELECTION FREQUENCY (%) BY DRIFT TYPE")
    print("=" * 80)

    detectors = tracker.portfolio.detector_names
    print(f"\n{'Detector':<15s} {'Sudden':>10s} {'Gradual':>10s} {'Increm.':>10s} {'Normal':>10s}")
    print("-" * 60)

    for detector in detectors:
        row = f"{detector:<15s}"
        for dtype in DRIFT_TYPES:
            freq = frequencies[dtype].get(detector, 0.0)
            row += f" {freq:>9.1f}"
        print(row)

    # Restructure frequencies for JSON output
    results_by_detector = {}
    for detector in detectors:
        results_by_detector[detector] = {}
        for dtype in DRIFT_TYPES:
            results_by_detector[detector][dtype] = round(frequencies[dtype].get(detector, 0.0), 1)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "experiment2_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'Detector Selection Patterns',
            'dataset_info': DATASET_INFO,
            'window_size': window_size,
            'detectors': detectors,
            'drift_types': DRIFT_TYPES,
            'results': results_by_detector
        }, f, indent=4)

    print(f"\nResults saved to: {output_file}")

    return results_by_detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment 2: Detector Selection Patterns')
    parser.add_argument('--data_dir', type=str, default=None,
                        help=f"Dataset directory (default: {DATASET_INFO['path']})")
    parser.add_argument('--window_size', type=int, default=DEFAULT_WINDOW_SIZE,
                        help=f'Window size (default: {DEFAULT_WINDOW_SIZE})')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    run_experiment_2(
        data_dir=args.data_dir,
        window_size=args.window_size,
        seed=args.seed
    )
