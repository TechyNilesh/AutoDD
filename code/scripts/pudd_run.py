"""
PUDD Drift Detector Benchmark Script

PUDD (Prediction Uncertainty-based Drift Detector) uses the PU-index
(prediction uncertainty) to detect drift earlier than error rate-based methods.

Reference:
    Lu, P., Lu, J., Liu, A., & Zhang, G. (2025).
    Early concept drift detection via prediction uncertainty.
    In AAAI Conference on Artificial Intelligence (AAAI-25).

Usage:
    python scripts/pudd_run.py --dataset elec
    python scripts/pudd_run.py --dataset sea --window_size 100 --alpha 0.05
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
from PUDD import PUDD

import warnings
warnings.filterwarnings("ignore")


def run_pudd(dataset_name, run_count=None, seed=42, progress_interval=1000,
             output_dir="benchmark_results", verbose=False,
             window_size=100, n_bins=10, alpha=0.05):
    """
    Run PUDD drift detector on a dataset with prequential evaluation.

    Args:
        dataset_name: Name of the dataset
        run_count: Run identifier for multiple runs
        seed: Random seed
        progress_interval: Interval for progress updates
        output_dir: Directory to save results
        verbose: Enable verbose output
        window_size: Size of the sliding window
        n_bins: Number of bins for PU-index histogram
        alpha: Significance level for Chi-square test
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running PUDD on {dataset_name}")
    print(f"{'='*60}")

    try:
        # Initialize model and drift detector
        model = naive_bayes.GaussianNB()
        detector = PUDD(window_size=window_size, n_bins=n_bins, alpha=alpha)

        # Initialize metrics
        accuracy = metrics.Accuracy()
        f1 = metrics.MacroF1()

        # Track metrics
        processed_instances = 0
        detected_drifts = []
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

            # Get prediction probabilities for PU-index
            y_proba = model.predict_proba_one(x)

            # Update metrics
            if y_pred is not None and y_proba:
                accuracy.update(y, y_pred)
                f1.update(y, y_pred)

                # Compute PU-index: 1 - probability of true class
                prob_true = y_proba.get(y, 0.0)
                pu_index = 1 - prob_true
                is_correct = (y_pred == y)

                # Update drift detector
                detector.update(pu_index, is_correct)

                # Check for drift
                if detector.drift_detected:
                    detected_drifts.append(processed_instances)
                    if verbose:
                        print(f"  Drift detected at instance {processed_instances}")
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
            "model": "PUDD",
            "detector_params": {
                "window_size": window_size,
                "n_bins": n_bins,
                "alpha": alpha
            },
            "accuracy": final_accuracy,
            "f1": final_f1,
            "n_drifts": len(detected_drifts),
            "drift_points": detected_drifts,
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
            file_name = f"PUDD_{dataset_name}_{run_count}.json"
        else:
            file_name = f"PUDD_{dataset_name}.json"

        # Save results
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n{'='*60}")
        print(f"PUDD completed on {dataset_name}")
        print(f"{'='*60}")
        print(f"Final Accuracy: {final_accuracy:.4f}")
        print(f"Final F1: {final_f1:.4f}")
        print(f"Detected Drifts: {len(detected_drifts)}")
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
    parser = argparse.ArgumentParser(description='Run PUDD drift detector on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Progress update interval')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of histogram bins')
    parser.add_argument('--alpha', type=float, default=0.05, help='Chi-square significance level')

    args = parser.parse_args()

    run_pudd(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        window_size=args.window_size,
        n_bins=args.n_bins,
        alpha=args.alpha
    )
