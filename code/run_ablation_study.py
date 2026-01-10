"""
AutoDD Ablation Study

Self-contained script to run ablation experiments testing component contributions:
1. Full AutoDD - all 15 meta-features with LinUCB
2. w/o Temporal Features - only statistical + stationarity (10 features)
3. w/o Statistical Features - only temporal + stationarity (10 features)
4. w/o Stationarity Features - only statistical + temporal (10 features)
5. w/o LinUCB - random detector selection

Usage:
    python run_ablation_study.py
    python run_ablation_study.py --n_runs 10
    python run_ablation_study.py --datasets elec weather
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import psutil
from collections import deque
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from river import naive_bayes, metrics, stream
from AutoDD.meta_feature_extractor import MetaFeatureExtractor
from AutoDD.detector_portfolio import DetectorPortfolio
from AutoDD.linucb_agent import LinUCBAgent

import warnings
warnings.filterwarnings("ignore")


# Datasets
SYNTHETIC_DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg']
REALWORLD_DATASETS = ['elec', 'weather', 'spam', 'airlines', 'poker']
ALL_DATASETS = SYNTHETIC_DATASETS + REALWORLD_DATASETS

# Feature indices
STATISTICAL_INDICES = [0, 1, 2, 3, 4]
STATIONARITY_INDICES = [5, 6, 7, 8, 9]
TEMPORAL_INDICES = [10, 11, 12, 13, 14]

# Ablation configurations
ABLATION_CONFIGS = {
    'full': {
        'name': 'Full AutoDD',
        'feature_indices': list(range(15)),
        'use_linucb': True
    },
    'no_temporal': {
        'name': 'w/o Temporal',
        'feature_indices': STATISTICAL_INDICES + STATIONARITY_INDICES,
        'use_linucb': True
    },
    'no_statistical': {
        'name': 'w/o Statistical',
        'feature_indices': STATIONARITY_INDICES + TEMPORAL_INDICES,
        'use_linucb': True
    },
    'no_stationarity': {
        'name': 'w/o Stationarity',
        'feature_indices': STATISTICAL_INDICES + TEMPORAL_INDICES,
        'use_linucb': True
    },
    'random': {
        'name': 'w/o LinUCB (Random)',
        'feature_indices': list(range(15)),
        'use_linucb': False
    }
}


class AblationAutoDD:
    """AutoDD with ablation support."""

    def __init__(self, window_size=100, alpha=1.0, feature_indices=None, use_linucb=True):
        self.window_size = window_size
        self.alpha = alpha
        self.feature_indices = feature_indices if feature_indices else list(range(15))
        self.use_linucb = use_linucb

        self.meta_extractor = MetaFeatureExtractor(window_size=window_size)
        self.portfolio = DetectorPortfolio()

        self.n_arms = len(self.portfolio)
        self.n_features = len(self.feature_indices)

        if use_linucb:
            self.agent = LinUCBAgent(n_arms=self.n_arms, n_features=self.n_features, alpha=alpha)
        else:
            self.agent = None

        self.arm_to_detector = {i: name for i, name in enumerate(self.portfolio.detector_names)}

        self._drift_detected = False
        self.current_detector = None
        self.current_arm = None
        self._n_detections = 0

        self.accuracy_buffer = deque(maxlen=window_size)
        self.prev_window_accuracy = None
        self.selection_history = []

        self._n_instances = 0
        self._active_detector_name = None
        self._active_detector = None
        self._last_meta_features = None

    def _select_features(self, full_features):
        return full_features[self.feature_indices]

    def update(self, x, y_true, y_pred):
        self._n_instances += 1
        self._drift_detected = False

        correct = 1.0 if y_true == y_pred else 0.0
        self.accuracy_buffer.append(correct)
        error = 0.0 if y_true == y_pred else 1.0

        self.meta_extractor.update(x, y_true, y_pred)

        if self._active_detector is not None:
            self._active_detector.update(error)
            if self._active_detector.drift_detected:
                self._drift_detected = True
                self._n_detections += 1

                if self._last_meta_features is not None and self.use_linucb:
                    reward = self._compute_reward()
                    self.agent.update(self.current_arm, self._last_meta_features, reward)

                self.portfolio.reset_detector(self._active_detector_name)
                self._active_detector = self.portfolio.get_detector(self._active_detector_name)

        if self._n_instances % self.window_size == 0:
            self._process_window()

        return self

    def _process_window(self):
        if not self.meta_extractor.is_ready():
            if self._active_detector is None:
                first_detector = self.portfolio.detector_names[0]
                self._active_detector_name = first_detector
                self._active_detector = self.portfolio.get_detector(first_detector)
                self.current_detector = first_detector
                self.current_arm = 0
            return

        full_features = self.meta_extractor.get_features()
        meta_features = self._select_features(full_features)

        if self._last_meta_features is not None and self.current_arm is not None and self.use_linucb:
            reward = self._compute_reward()
            self.agent.update(self.current_arm, self._last_meta_features, reward)

        if self.use_linucb:
            selected_arm = self.agent.select_arm(meta_features)
        else:
            selected_arm = np.random.randint(0, self.n_arms)

        selected_detector = self.arm_to_detector[selected_arm]
        self.current_arm = selected_arm
        self.current_detector = selected_detector
        self.selection_history.append(selected_detector)

        self._active_detector_name = selected_detector
        self.portfolio.reset_detector(selected_detector)
        self._active_detector = self.portfolio.get_detector(selected_detector)

        self._last_meta_features = meta_features.copy()
        self.prev_window_accuracy = np.mean(list(self.accuracy_buffer))

    def _compute_reward(self):
        if len(self.accuracy_buffer) == 0 or self.prev_window_accuracy is None:
            return 0.0

        current_accuracy = np.mean(list(self.accuracy_buffer))
        accuracy_drop = self.prev_window_accuracy - current_accuracy
        significant_drop = accuracy_drop > 0.05

        if self._drift_detected:
            reward = 0.5 + min(accuracy_drop * 2, 0.5) if significant_drop else -0.3
        else:
            reward = -0.5 - min(accuracy_drop, 0.5) if significant_drop else 0.1

        return float(np.clip(reward, -1.0, 1.0))

    @property
    def drift_detected(self):
        return self._drift_detected

    @property
    def n_detections(self):
        return self._n_detections

    def get_selection_distribution(self):
        if not self.selection_history:
            return {}
        counts = {}
        for name in self.portfolio.detector_names:
            counts[name] = self.selection_history.count(name) / len(self.selection_history)
        return counts


def run_single_ablation(dataset_name, config_name, run_count=None, seed=42, output_dir="ablation_results"):
    """Run single ablation experiment."""

    np.random.seed(seed)
    config = ABLATION_CONFIGS[config_name]

    try:
        model = naive_bayes.GaussianNB()
        detector = AblationAutoDD(
            window_size=100,
            alpha=1.0,
            feature_indices=config['feature_indices'],
            use_linucb=config['use_linucb']
        )

        accuracy = metrics.Accuracy()
        f1 = metrics.MacroF1()

        processed_instances = 0
        detected_drifts = []

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        dataset_path = f"datasets/{dataset_name}.csv"

        for x, y in stream.iter_csv(dataset_path, target='class', converters={'class': str}):
            x = {k: float(v) for k, v in x.items()}
            y_pred = model.predict_one(x)

            if y_pred is not None:
                accuracy.update(y, y_pred)
                f1.update(y, y_pred)

            detector.update(x, y, y_pred if y_pred is not None else y)

            if detector.drift_detected:
                detected_drifts.append(processed_instances)
                model = naive_bayes.GaussianNB()

            model.learn_one(x, y)
            processed_instances += 1

        final_accuracy = accuracy.get()
        final_f1 = f1.get()
        total_runtime = time.time() - start_time
        total_memory = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)

        results = {
            "dataset": dataset_name,
            "config": config_name,
            "config_name": config['name'],
            "n_features": len(config['feature_indices']),
            "use_linucb": config['use_linucb'],
            "accuracy": final_accuracy,
            "f1": final_f1,
            "n_drifts": len(detected_drifts),
            "total_runtime": total_runtime,
            "total_memory_mb": total_memory,
            "n_instances": processed_instances,
            "selection_distribution": detector.get_selection_distribution()
        }

        if run_count is not None:
            results["run_count"] = run_count
            results["seed"] = seed
            file_name = f"ablation_{config_name}_{dataset_name}_{run_count}.json"
        else:
            file_name = f"ablation_{config_name}_{dataset_name}.json"

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"  {config['name']:25s} | {dataset_name:10s} | Acc: {final_accuracy:.4f} | F1: {final_f1:.4f} | Drifts: {len(detected_drifts):4d}")

        return results

    except Exception as e:
        print(f"  ERROR: {config_name} on {dataset_name}: {str(e)}")
        return None


def run_ablation_study(datasets=None, n_runs=1, output_dir="ablation_results", max_workers=None):
    """Run complete ablation study."""

    if datasets is None:
        datasets = ALL_DATASETS

    os.makedirs(output_dir, exist_ok=True)

    total_experiments = len(datasets) * len(ABLATION_CONFIGS) * n_runs

    print("=" * 80)
    print("AutoDD Ablation Study")
    print("=" * 80)
    print(f"Datasets: {len(datasets)}")
    print(f"Configurations: {len(ABLATION_CONFIGS)}")
    print(f"Runs per config: {n_runs}")
    print(f"Total experiments: {total_experiments}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()

    tasks = []
    for run in range(1, n_runs + 1):
        for dataset in datasets:
            for config_name in ABLATION_CONFIGS:
                seed = run * 42
                run_count = run if n_runs > 1 else None
                tasks.append((dataset, config_name, run_count, seed, output_dir))

    start_time = time.time()

    if max_workers and max_workers > 1:
        print(f"Running with {max_workers} parallel workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_ablation, *task) for task in tasks]
            concurrent.futures.wait(futures)
    else:
        print("Running sequentially...")
        for task in tasks:
            run_single_ablation(*task)

    total_time = time.time() - start_time

    print()
    print("=" * 80)
    print(f"Ablation Study Complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}/")
    print("=" * 80)

    # Generate summary
    generate_summary(output_dir, datasets)


def generate_summary(output_dir, datasets):
    """Generate summary of ablation results."""

    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)

    # Collect results
    summary = {config: {'accuracy': [], 'f1': []} for config in ABLATION_CONFIGS}

    for filename in os.listdir(output_dir):
        if filename.endswith('.json') and filename.startswith('ablation_'):
            with open(os.path.join(output_dir, filename), 'r') as f:
                result = json.load(f)
                config = result.get('config')
                if config in summary:
                    summary[config]['accuracy'].append(result['accuracy'])
                    summary[config]['f1'].append(result['f1'])

    # Print summary table
    print(f"\n{'Configuration':<30s} {'Avg Accuracy':>12s} {'Avg F1':>10s} {'Delta':>10s}")
    print("-" * 65)

    full_acc = np.mean(summary['full']['accuracy']) if summary['full']['accuracy'] else 0

    for config_name, config in ABLATION_CONFIGS.items():
        if summary[config_name]['accuracy']:
            avg_acc = np.mean(summary[config_name]['accuracy'])
            avg_f1 = np.mean(summary[config_name]['f1'])
            delta = avg_acc - full_acc
            delta_str = f"{delta:+.2%}" if config_name != 'full' else "--"
            print(f"{config['name']:<30s} {avg_acc:>11.2%} {avg_f1:>10.4f} {delta_str:>10s}")

    # Save summary
    summary_file = os.path.join(output_dir, "ablation_summary.json")
    summary_data = {
        config: {
            'name': ABLATION_CONFIGS[config]['name'],
            'avg_accuracy': np.mean(data['accuracy']) if data['accuracy'] else None,
            'avg_f1': np.mean(data['f1']) if data['f1'] else None,
            'n_results': len(data['accuracy'])
        }
        for config, data in summary.items()
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AutoDD Ablation Study')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help=f'Datasets to use (default: all). Available: {ALL_DATASETS}')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of runs per configuration (default: 1)')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                        help='Output directory (default: ablation_results)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Max parallel workers (default: sequential)')

    args = parser.parse_args()

    run_ablation_study(
        datasets=args.datasets,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
