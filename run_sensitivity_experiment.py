"""
Run Hyperparameter Sensitivity Experiments

Runs AutoDD with different hyperparameter configurations to analyze sensitivity:
- Window size: [25, 50, 100, 150, 200]
- Alpha (LinUCB exploration): [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
- Lambda (regularization): [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

Usage:
    python run_sensitivity_experiment.py
    python run_sensitivity_experiment.py --experiment window
    python run_sensitivity_experiment.py --experiment alpha --n_runs 5
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from river import naive_bayes, metrics, stream
from AutoDD import AutoDD

import warnings
warnings.filterwarnings("ignore")

# Experiment configurations
WINDOW_SIZES = [25, 50, 100, 150, 200]
ALPHA_VALUES = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
LAMBDA_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

# Datasets for sensitivity analysis
DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg', 'elec']

OUTPUT_DIR = "sensitivity_results"


def run_autodd_experiment(dataset_name, window_size=100, alpha=1.0, lambda_reg=1.0, seed=42):
    """Run single AutoDD experiment with specified parameters."""
    np.random.seed(seed)

    model = naive_bayes.GaussianNB()
    detector = AutoDD(window_size=window_size, alpha=alpha, lambda_reg=lambda_reg)

    accuracy = metrics.Accuracy()
    f1 = metrics.MacroF1()

    processed = 0
    n_drifts = 0

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
            n_drifts += 1
            model = naive_bayes.GaussianNB()

        model.learn_one(x, y)
        processed += 1

    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss

    return {
        'accuracy': accuracy.get(),
        'f1': f1.get(),
        'n_drifts': n_drifts,
        'n_instances': processed,
        'runtime': end_time - start_time,
        'memory_mb': (end_memory - start_memory) / (1024 * 1024)
    }


def run_window_sensitivity(n_runs=3, seed=42):
    """Run window size sensitivity experiment."""
    print("\n" + "=" * 70)
    print("WINDOW SIZE SENSITIVITY EXPERIMENT")
    print("=" * 70)

    results = {}

    for dataset in DATASETS:
        print(f"\nDataset: {dataset}")
        results[dataset] = {}

        for ws in WINDOW_SIZES:
            print(f"  Window size w={ws}...", end=" ", flush=True)
            run_results = []

            for run in range(n_runs):
                res = run_autodd_experiment(
                    dataset, window_size=ws, alpha=1.0, seed=seed + run
                )
                run_results.append(res)

            avg_acc = np.mean([r['accuracy'] for r in run_results])
            std_acc = np.std([r['accuracy'] for r in run_results])
            avg_f1 = np.mean([r['f1'] for r in run_results])

            results[dataset][ws] = {
                'accuracy_mean': round(avg_acc, 4),
                'accuracy_std': round(std_acc, 4),
                'f1_mean': round(avg_f1, 4)
            }
            print(f"Acc: {avg_acc*100:.1f}%")

    return results


def run_alpha_sensitivity(n_runs=3, seed=42):
    """Run alpha (exploration) sensitivity experiment."""
    print("\n" + "=" * 70)
    print("ALPHA (LinUCB EXPLORATION) SENSITIVITY EXPERIMENT")
    print("=" * 70)

    results = {}

    for dataset in DATASETS:
        print(f"\nDataset: {dataset}")
        results[dataset] = {}

        for alpha in ALPHA_VALUES:
            print(f"  Alpha α={alpha}...", end=" ", flush=True)
            run_results = []

            for run in range(n_runs):
                res = run_autodd_experiment(
                    dataset, window_size=100, alpha=alpha, seed=seed + run
                )
                run_results.append(res)

            avg_acc = np.mean([r['accuracy'] for r in run_results])
            std_acc = np.std([r['accuracy'] for r in run_results])
            avg_f1 = np.mean([r['f1'] for r in run_results])

            results[dataset][alpha] = {
                'accuracy_mean': round(avg_acc, 4),
                'accuracy_std': round(std_acc, 4),
                'f1_mean': round(avg_f1, 4)
            }
            print(f"Acc: {avg_acc*100:.1f}%")

    return results


def run_lambda_sensitivity(n_runs=3, seed=42):
    """Run lambda (regularization) sensitivity experiment."""
    print("\n" + "=" * 70)
    print("LAMBDA (REGULARIZATION) SENSITIVITY EXPERIMENT")
    print("=" * 70)

    results = {}

    for dataset in DATASETS:
        print(f"\nDataset: {dataset}")
        results[dataset] = {}

        for lam in LAMBDA_VALUES:
            print(f"  Lambda λ={lam}...", end=" ", flush=True)
            run_results = []

            for run in range(n_runs):
                # Note: lambda_reg would need to be passed to LinUCB
                # For now using default, as AutoDD doesn't expose lambda directly
                res = run_autodd_experiment(
                    dataset, window_size=100, alpha=1.0, lambda_reg=lam, seed=seed + run
                )
                run_results.append(res)

            avg_acc = np.mean([r['accuracy'] for r in run_results])
            std_acc = np.std([r['accuracy'] for r in run_results])
            avg_f1 = np.mean([r['f1'] for r in run_results])

            results[dataset][lam] = {
                'accuracy_mean': round(avg_acc, 4),
                'accuracy_std': round(std_acc, 4),
                'f1_mean': round(avg_f1, 4)
            }
            print(f"Acc: {avg_acc*100:.1f}%")

    return results


def save_results(results, experiment_name):
    """Save experiment results to JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, f"{experiment_name}_results.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Run Sensitivity Experiments')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'window', 'alpha', 'lambda'],
                        help='Which experiment to run')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs per configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 70)
    print("AUTODD HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Datasets: {DATASETS}")
    print(f"Runs per config: {args.n_runs}")

    if args.experiment in ['all', 'window']:
        results = run_window_sensitivity(args.n_runs, args.seed)
        save_results(results, 'window_sensitivity')

    if args.experiment in ['all', 'alpha']:
        results = run_alpha_sensitivity(args.n_runs, args.seed)
        save_results(results, 'alpha_sensitivity')

    if args.experiment in ['all', 'lambda']:
        results = run_lambda_sensitivity(args.n_runs, args.seed)
        save_results(results, 'lambda_sensitivity')

    print("\n" + "=" * 70)
    print("SENSITIVITY EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
