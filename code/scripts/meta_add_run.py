"""
Meta-ADD Drift Detector Benchmark Script

Meta-ADD uses meta-learning with prototypical neural networks to detect
and classify different types of concept drift.

Usage:
    python scripts/meta_add_run.py --dataset elec
    python scripts/meta_add_run.py --dataset sea --run_count 1 --seed 42
"""

import time
import psutil
import json
import os
import sys
import argparse
import traceback

# Add parent directory to path for MetaADD import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from river import naive_bayes, metrics, stream

from MetaADD import MetaADD, train_meta_detector, load_pretrained_model, quick_pretrain
from MetaADD.trainer import generate_synthetic_training_data

import warnings
warnings.filterwarnings("ignore")


def get_pretrained_model(seed=42, use_cached=True):
    """
    Get a pre-trained Meta-ADD model.

    First tries to load a pre-trained model from disk.
    If not available, trains a new model with quick_pretrain.

    Parameters
    ----------
    seed : int
        Random seed for training
    use_cached : bool
        Whether to try loading a cached model first
    """
    # Try to load pre-trained model first
    if use_cached:
        try:
            print("Loading pre-trained model...")
            model = load_pretrained_model()
            print("Pre-trained model loaded successfully!")
            return model
        except (FileNotFoundError, Exception) as e:
            print(f"No pre-trained model found: {e}")
            print("Training new model...")

    # Fall back to quick training
    print("Running quick pre-training (this may take a few minutes)...")
    model = quick_pretrain(verbose=False)
    print("Pre-training complete!")

    return model


def run_meta_add(dataset_name, run_count=None, seed=42, progress_interval=1000,
                 output_dir="benchmark_results", verbose=False,
                 window_size=50, n_meta_features=100, pretrain=True):  # Match pretraining: n=50, l=100
    """
    Run Meta-ADD drift detector on a dataset with prequential evaluation.

    Args:
        dataset_name: Name of the dataset
        run_count: Run identifier for multiple runs
        seed: Random seed
        progress_interval: Interval for progress updates
        output_dir: Directory to save results
        verbose: Enable verbose output
        window_size: Window size for meta-feature extraction
        n_meta_features: Number of meta-features
        pretrain: Whether to use pre-trained model
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running Meta-ADD on {dataset_name}")
    print(f"{'='*60}")

    try:
        # Get pre-trained model if requested
        pretrained_model = None
        if pretrain:
            pretrained_model = get_pretrained_model(seed)

        # Initialize model and drift detector
        model = naive_bayes.GaussianNB()
        detector = MetaADD(
            window_size=window_size,
            n_meta_features=n_meta_features,
            embedding_dim=128,
            uncertainty_threshold=0.5,
            pretrained_model=pretrained_model
        )

        # Initialize metrics
        accuracy = metrics.Accuracy()
        f1 = metrics.MacroF1()

        # Track metrics
        processed_instances = 0
        detected_drifts = []
        drift_types = []
        accuracy_scores = []
        f1_scores = []
        times = []
        memory_usages = []

        # Start timing and memory tracking
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Load and process dataset
        dataset_path = f"datasets/{dataset_name}.csv"

        print(f"Processing {dataset_path}...")

        for x, y in stream.iter_csv(dataset_path, target='class', converters={'class': str}):
            # Convert features to float
            x = {k: float(v) for k, v in x.items()}
            # Predict before learning (prequential)
            y_pred = model.predict_one(x)

            # Update metrics
            if y_pred is not None:
                accuracy.update(y, y_pred)
                f1.update(y, y_pred)

                # Update drift detector
                detector.update(y, y_pred)

                # Check for drift
                if detector.drift_detected:
                    detected_drifts.append(processed_instances)
                    drift_info = detector.get_drift_info()
                    drift_types.append(drift_info['drift_type'])

                    if verbose:
                        print(f"  Drift detected at instance {processed_instances}: "
                              f"{drift_info['drift_type']} (conf: {drift_info['confidence']:.3f})")

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

        # Count drift types
        drift_type_counts = {}
        for dt in drift_types:
            drift_type_counts[dt] = drift_type_counts.get(dt, 0) + 1

        # Store results
        results = {
            "dataset": dataset_name,
            "model": "MetaADD",
            "detector_params": {
                "window_size": window_size,
                "n_meta_features": n_meta_features,
                "pretrained": pretrain
            },
            "accuracy": final_accuracy,
            "f1": final_f1,
            "n_drifts": len(detected_drifts),
            "drift_points": detected_drifts,
            "drift_types": drift_types,
            "drift_type_counts": drift_type_counts,
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
            file_name = f"MetaADD_{dataset_name}_{run_count}.json"
        else:
            file_name = f"MetaADD_{dataset_name}.json"

        # Save results
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\n{'='*60}")
        print(f"Meta-ADD completed on {dataset_name}")
        print(f"{'='*60}")
        print(f"Final Accuracy: {final_accuracy:.4f}")
        print(f"Final F1: {final_f1:.4f}")
        print(f"Detected Drifts: {len(detected_drifts)}")
        print(f"Drift Type Counts: {drift_type_counts}")
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
    parser = argparse.ArgumentParser(description='Run Meta-ADD drift detector on a dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--run_count', type=int, default=None, help='Run count identifier')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--progress_interval', type=int, default=1000, help='Progress update interval')
    parser.add_argument('--output_dir', type=str, default="benchmark_results", help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--window_size', type=int, default=50, help='Window size (match pretraining: n=50)')
    parser.add_argument('--n_meta_features', type=int, default=100, help='Number of meta-features (paper: l=100)')
    parser.add_argument('--no_pretrain', action='store_true', help='Disable pre-training')

    args = parser.parse_args()

    run_meta_add(
        dataset_name=args.dataset,
        run_count=args.run_count,
        seed=args.seed,
        progress_interval=args.progress_interval,
        output_dir=args.output_dir,
        verbose=args.verbose,
        window_size=args.window_size,
        n_meta_features=args.n_meta_features,
        pretrain=not args.no_pretrain
    )
