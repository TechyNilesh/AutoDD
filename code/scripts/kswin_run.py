"""
KSWIN Drift Detector Benchmark Script

KSWIN (Kolmogorov-Smirnov Windowing) uses the Kolmogorov-Smirnov test
to detect concept drift by comparing distributions in sliding windows.

Usage:
    python scripts/kswin_run.py --dataset elec
    python scripts/kswin_run.py --dataset sea --run_count 1 --seed 42
    python scripts/kswin_run.py --dataset elec --base_model hat
"""

import time
import psutil
import json
import os
import sys
import argparse
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from river import drift, metrics, stream

from scripts.utils import get_base_model, get_base_model_display_name

import warnings
warnings.filterwarnings("ignore")


def run_kswin(dataset_name, run_count=None, seed=42, progress_interval=1000,
              output_dir="benchmark_results", verbose=False,
              alpha=0.005, window_size=100, stat_size=30, base_model='gaussiannb'):
    """
    Run KSWIN drift detector on a dataset with prequential evaluation.

    Args:
        dataset_name: Name of the dataset
        run_count: Run identifier for multiple runs
        seed: Random seed
        progress_interval: Interval for progress updates
        output_dir: Directory to save results
        verbose: Enable verbose output
        alpha: Probability for the KS test
        window_size: Size of the sliding window
        stat_size: Size of the statistic window
        base_model: Base classifier to use ('gaussiannb' or 'hat')
    """
    os.makedirs(output_dir, exist_ok=True)

    base_model_name = get_base_model_display_name(base_model)
    print(f"\n{'='*60}")
    print(f"Running KSWIN on {dataset_name} (Base: {base_model_name})")
    print(f"{'='*60}")

    try:
        # Initialize model and drift detector
        model = get_base_model(base_model)
        detector = drift.KSWIN(
            alpha=alpha,
            window_size=window_size,
            stat_size=stat_size,
            seed=seed
        )

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
                    model = get_base_model(base_model)

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
            "model": "KSWIN",
            "base_model": base_model_name,
            "detector_params": {
                "alpha": alpha,
                "window_size": window_size,
                "stat_size": stat_size
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
            file_name = f"KSWIN_{base_model_name}_{dataset_name}_{run_count}.json"
        else:
            file_name = f"KSWIN_{base_model_name}_{dataset_name}.json"

        # Save results
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n{'='*60}")
        print(f"KSWIN completed on {dataset_name}")
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
    parser = argparse.ArgumentParser(description='Run KSWIN drift detector on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Progress update interval')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--alpha', type=float, default=0.005, help='KS test alpha')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--stat_size', type=int, default=30, help='Statistic window size')
    parser.add_argument('--base_model', type=str, default='gaussiannb',
                        choices=['gaussiannb', 'hat'],
                        help='Base classifier to use (default: gaussiannb)')

    args = parser.parse_args()

    run_kswin(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        alpha=args.alpha,
        window_size=args.window_size,
        stat_size=args.stat_size,
        base_model=args.base_model
    )
