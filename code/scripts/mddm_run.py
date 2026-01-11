"""
MDDM Drift Detector Benchmark Script

MDDM (McDiarmid Drift Detection Method) uses McDiarmid's inequality
with weighted sliding windows to detect concept drift.

Three variants available:
- MDDM_A: Arithmetic (linear) weighting
- MDDM_G: Geometric (base-2 exponential) weighting
- MDDM_E: Euler (base-e exponential) weighting

Reference:
    Pesaranghader, A., Viktor, H., & Paquet, E. (2018).
    McDiarmid drift detection methods for evolving data streams.
    In IJCNN 2018 (pp. 1-9). IEEE.

Usage:
    python scripts/mddm_run.py --dataset elec
    python scripts/mddm_run.py --dataset sea --variant G --n 100
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
from MDDM import MDDM, MDDM_A, MDDM_G, MDDM_E

import warnings
warnings.filterwarnings("ignore")


def run_mddm(dataset_name, run_count=None, seed=42, progress_interval=1000,
             output_dir="benchmark_results", verbose=False,
             variant='A', n=100, delta=0.000001):
    """
    Run MDDM drift detector on a dataset with prequential evaluation.

    Args:
        dataset_name: Name of the dataset
        run_count: Run identifier for multiple runs
        seed: Random seed
        progress_interval: Interval for progress updates
        output_dir: Directory to save results
        verbose: Enable verbose output
        variant: MDDM variant ('A', 'G', or 'E')
        n: Window size
        delta: Confidence parameter for McDiarmid's bound
    """
    os.makedirs(output_dir, exist_ok=True)

    detector_name = f"MDDM_{variant}"
    print(f"\n{'='*60}")
    print(f"Running {detector_name} on {dataset_name}")
    print(f"{'='*60}")

    try:
        # Initialize model and drift detector
        model = naive_bayes.GaussianNB()

        if variant == 'A':
            detector = MDDM_A(n=n, delta=delta)
        elif variant == 'G':
            detector = MDDM_G(n=n, delta=delta)
        elif variant == 'E':
            detector = MDDM_E(n=n, delta=delta)
        else:
            detector = MDDM(n=n, delta=delta, weighting=variant.lower())

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
            "model": detector_name,
            "detector_params": {
                "variant": variant,
                "n": n,
                "delta": delta
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
            file_name = f"{detector_name}_{dataset_name}_{run_count}.json"
        else:
            file_name = f"{detector_name}_{dataset_name}.json"

        # Save results
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n{'='*60}")
        print(f"{detector_name} completed on {dataset_name}")
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
    parser = argparse.ArgumentParser(description='Run MDDM drift detector on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Progress update interval')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--variant', type=str, default='A', choices=['A', 'G', 'E'],
                        help='MDDM variant: A=Arithmetic, G=Geometric, E=Euler')
    parser.add_argument('--n', type=int, default=100, help='Window size')
    parser.add_argument('--delta', type=float, default=0.000001, help='Confidence parameter')

    args = parser.parse_args()

    run_mddm(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        variant=args.variant,
        n=args.n,
        delta=args.delta
    )
