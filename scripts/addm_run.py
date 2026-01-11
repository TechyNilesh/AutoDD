"""
ADDM Drift Detector Benchmark Script

ADDM (Autoregressive-Based Drift Detection Method) models the error rate
as a time series using SETAR models to detect regime changes.

Reference:
    Mayaki, M. Z. A., & Riveill, M. (2023).
    Autoregressive based drift detection method.
    arXiv:2203.04769v2

Usage:
    python scripts/addm_run.py --dataset elec
    python scripts/addm_run.py --dataset sea --lag 5 --threshold 2.0
"""

import time
import psutil
import json
import os
import argparse
import traceback
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from river import naive_bayes, metrics, stream
from ADDM import ADDM

import warnings
warnings.filterwarnings("ignore")


def run_addm(dataset_name, run_count=None, seed=42, progress_interval=1000,
             output_dir="benchmark_results", verbose=False,
             window_size=100, lag=5, threshold=2.0):
    """
    Run ADDM drift detector on a dataset with prequential evaluation.

    Args:
        dataset_name: Name of the dataset
        run_count: Run identifier for multiple runs
        seed: Random seed
        progress_interval: Interval for progress updates
        output_dir: Directory to save results
        verbose: Enable verbose output
        window_size: Size of the sliding window
        lag: Autoregressive lag order
        threshold: Number of standard deviations for drift detection
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running ADDM on {dataset_name}")
    print(f"{'='*60}")

    try:
        # Initialize model and drift detector
        model = naive_bayes.GaussianNB()
        detector = ADDM(window_size=window_size, lag=lag, threshold=threshold)

        # Initialize metrics
        accuracy = metrics.Accuracy()
        f1 = metrics.MacroF1()

        # Track metrics
        processed_instances = 0
        detected_drifts = []
        drift_severities = []
        accuracy_scores = []
        f1_scores = []
        times = []
        memory_usages = []

        # Start timing and memory tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Load and process dataset
        dataset_path = f"datasets/{dataset_name}.csv"

        for x, y in stream.iter_csv(dataset_path, target='class', converters={'class': str}):
            # Convert features to float
            x = {k: float(v) for k, v in x.items()}

            # Predict before learning (prequential)
            y_pred = model.predict_one(x)

            # Update metrics
            if y_pred is not None:
                accuracy.update(y, y_pred)
                f1.update(y, y_pred)

                # Update drift detector with prediction correctness
                error = 0 if y_pred == y else 1
                detector.update(error)

                # Check for drift
                if detector.drift_detected:
                    detected_drifts.append(processed_instances)
                    severity = detector.get_drift_severity()
                    if severity is not None:
                        drift_severities.append(severity)
                    if verbose:
                        print(f"  Drift detected at instance {processed_instances}"
                              f" (severity: {severity:.3f})" if severity else "")
                    # Reset model on drift
                    model = naive_bayes.GaussianNB()

            # Learn from instance
            model.learn_one(x, y)
            processed_instances += 1

            # Progress update
            if processed_instances % progress_interval == 0:
                current_acc = accuracy.get()
                current_f1 = f1.get()
                accuracy_scores.append(current_acc)
                f1_scores.append(current_f1)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                memory_usage = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)
                memory_usages.append(memory_usage)

                if verbose:
                    print(f"  Processed {processed_instances} instances, "
                          f"Accuracy: {current_acc:.4f}, "
                          f"Drifts: {len(detected_drifts)}, "
                          f"Time: {elapsed_time:.2f}s, "
                          f"Memory: {memory_usage:.2f}MB")

        # Calculate final metrics
        final_accuracy = accuracy.get()
        final_f1 = f1.get()
        total_runtime = time.time() - start_time
        total_memory = abs((psutil.Process().memory_info().rss - start_memory) / 1024 / 1024)

        # Store results
        results = {
            "dataset": dataset_name,
            "model": "ADDM",
            "detector_params": {
                "window_size": window_size,
                "lag": lag,
                "threshold": threshold
            },
            "accuracy": final_accuracy,
            "f1": final_f1,
            "n_drifts": len(detected_drifts),
            "drift_points": detected_drifts,
            "drift_severities": drift_severities,
            "total_runtime": total_runtime,
            "total_memory_mb": total_memory,
            "n_instances": processed_instances,
            "window_metrics": {
                "interval": progress_interval,
                "accuracy_scores": accuracy_scores,
                "f1_scores": f1_scores,
                "runtimes": times,
                "memory_usages": memory_usages
            }
        }

        if run_count is not None:
            results["run_count"] = run_count
            results["seed"] = seed
            file_name = f"ADDM_{dataset_name}_{run_count}.json"
        else:
            file_name = f"ADDM_{dataset_name}.json"

        # Save results
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n{'='*60}")
        print(f"ADDM completed on {dataset_name}")
        print(f"{'='*60}")
        print(f"Final Accuracy: {final_accuracy:.4f}")
        print(f"Final F1: {final_f1:.4f}")
        print(f"Detected Drifts: {len(detected_drifts)}")
        if drift_severities:
            print(f"Average Drift Severity: {sum(drift_severities)/len(drift_severities):.3f}")
        print(f"Runtime: {total_runtime:.2f} seconds")
        print(f"Memory: {total_memory:.2f} MB")
        print(f"Processed {processed_instances} instances")
        print(f"Results saved to {output_file}")

        return results

    except Exception as e:
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'stack_trace': traceback.format_exc()
        }
        print(f"\nError: {error_info['error_type']}: {error_info['error_message']}")
        print(error_info['stack_trace'])
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ADDM drift detector on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Progress update interval')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--lag', type=int, default=5, help='Autoregressive lag order')
    parser.add_argument('--threshold', type=float, default=2.0, help='Drift detection threshold')

    args = parser.parse_args()

    run_addm(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        window_size=args.window_size,
        lag=args.lag,
        threshold=args.threshold
    )
