"""
Run 2D Hyperparameter Sensitivity Experiments

Runs AutoDD with a grid of TWO hyperparameter combinations to generate
data suitable for 3D surface plots (like the OASF paper).

This creates a 2D grid of parameter combinations and measures error rates
for each combination, enabling visualization of parameter interactions.

Usage:
    python run_sensitivity_2d_experiment.py
    python run_sensitivity_2d_experiment.py --params alpha lambda --n_runs 5
    python run_sensitivity_2d_experiment.py --params window alpha --dataset sea
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

# Parameter value configurations (can be customized)
PARAM_VALUES = {
    'alpha': [0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
    'lambda': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    'window': [25, 50, 100, 150, 200]
}

PARAM_NAMES = {
    'alpha': 'Parameter α',
    'lambda': 'Parameter λ',
    'window': 'Window Size'
}

# Datasets for experiments
DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg', 'elec']

OUTPUT_DIR = "experiment_results"


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
    
    if not os.path.exists(dataset_path):
        print(f"  Warning: Dataset {dataset_path} not found, skipping...")
        return None
    
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
        'error_rate': 1 - accuracy.get(),
        'n_drifts': n_drifts,
        'n_instances': processed,
        'runtime': end_time - start_time,
        'memory_mb': (end_memory - start_memory) / (1024 * 1024)
    }


def run_2d_sensitivity(param1_name, param2_name, datasets=None, n_runs=3, seed=42):
    """
    Run 2D sensitivity experiment for two parameters.
    
    Args:
        param1_name: First parameter name ('alpha', 'lambda', or 'window')
        param2_name: Second parameter name
        datasets: List of datasets to evaluate (default: all)
        n_runs: Number of runs per configuration
        seed: Random seed
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*70}")
    print(f"2D SENSITIVITY: {param1_name.upper()} vs {param2_name.upper()}")
    print(f"{'='*70}")
    
    param1_values = PARAM_VALUES[param1_name]
    param2_values = PARAM_VALUES[param2_name]
    
    if datasets is None:
        datasets = DATASETS
    
    results = {
        'experiment': f'2D Sensitivity: {param1_name} x {param2_name}',
        'param1_name': PARAM_NAMES[param1_name],
        'param2_name': PARAM_NAMES[param2_name],
        'param1_key': param1_name,
        'param2_key': param2_name,
        'param1_values': param1_values,
        'param2_values': param2_values,
        'n_runs': n_runs,
        'datasets': {}
    }
    
    total_configs = len(param1_values) * len(param2_values)
    
    for dataset in datasets:
        print(f"\nDataset: {dataset.upper()}")
        print(f"  Grid: {len(param2_values)} x {len(param1_values)} = {total_configs} configurations")
        
        # Initialize error matrix: rows = param2, cols = param1
        error_matrix = []
        accuracy_matrix = []
        f1_matrix = []
        std_matrix = []
        
        config_count = 0
        
        for p2_val in param2_values:
            error_row = []
            acc_row = []
            f1_row = []
            std_row = []
            
            for p1_val in param1_values:
                config_count += 1
                
                # Build parameter dict
                params = {'window_size': 100, 'alpha': 1.0, 'lambda_reg': 1.0}
                
                if param1_name == 'alpha':
                    params['alpha'] = p1_val
                elif param1_name == 'lambda':
                    params['lambda_reg'] = p1_val
                elif param1_name == 'window':
                    params['window_size'] = p1_val
                
                if param2_name == 'alpha':
                    params['alpha'] = p2_val
                elif param2_name == 'lambda':
                    params['lambda_reg'] = p2_val
                elif param2_name == 'window':
                    params['window_size'] = p2_val
                
                print(f"  [{config_count}/{total_configs}] "
                      f"{param1_name}={p1_val}, {param2_name}={p2_val}...", 
                      end=" ", flush=True)
                
                run_results = []
                for run in range(n_runs):
                    res = run_autodd_experiment(
                        dataset,
                        window_size=params['window_size'],
                        alpha=params['alpha'],
                        lambda_reg=params['lambda_reg'],
                        seed=seed + run
                    )
                    if res is not None:
                        run_results.append(res)
                
                if run_results:
                    avg_error = np.mean([r['error_rate'] for r in run_results])
                    avg_acc = np.mean([r['accuracy'] for r in run_results])
                    std_acc = np.std([r['accuracy'] for r in run_results])
                    avg_f1 = np.mean([r['f1'] for r in run_results])
                    
                    error_row.append(round(avg_error, 4))
                    acc_row.append(round(avg_acc, 4))
                    std_row.append(round(std_acc, 4))
                    f1_row.append(round(avg_f1, 4))
                    
                    print(f"Error: {avg_error*100:.2f}%")
                else:
                    error_row.append(None)
                    acc_row.append(None)
                    std_row.append(None)
                    f1_row.append(None)
                    print("SKIPPED")
            
            error_matrix.append(error_row)
            accuracy_matrix.append(acc_row)
            f1_matrix.append(f1_row)
            std_matrix.append(std_row)
        
        results['datasets'][dataset] = {
            'error_matrix': error_matrix,
            'accuracy_matrix': accuracy_matrix,
            'f1_matrix': f1_matrix,
            'std_matrix': std_matrix
        }
        
        # Find optimal configuration
        error_array = np.array(error_matrix)
        if not np.all(error_array == None):
            min_idx = np.unravel_index(np.nanargmin(error_array), error_array.shape)
            optimal_p2 = param2_values[min_idx[0]]
            optimal_p1 = param1_values[min_idx[1]]
            min_error = error_array[min_idx]
            print(f"  Optimal: {param1_name}={optimal_p1}, {param2_name}={optimal_p2} "
                  f"(Error: {min_error*100:.2f}%)")
    
    return results


def save_results(results, filename):
    """Save experiment results to JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description='Run 2D Sensitivity Experiments')
    parser.add_argument('--params', nargs=2, default=['alpha', 'lambda'],
                        choices=['alpha', 'lambda', 'window'],
                        help='Two parameters to vary (default: alpha lambda)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to evaluate (default: all)')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs per configuration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    param1, param2 = args.params
    
    if param1 == param2:
        print("Error: Please specify two different parameters.")
        return
    
    print("=" * 70)
    print("AUTODD 2D HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Parameters: {param1} (X-axis) vs {param2} (Y-axis)")
    print(f"Datasets: {args.datasets or DATASETS}")
    print(f"Runs per config: {args.n_runs}")
    print(f"Total configurations per dataset: "
          f"{len(PARAM_VALUES[param1])} x {len(PARAM_VALUES[param2])} = "
          f"{len(PARAM_VALUES[param1]) * len(PARAM_VALUES[param2])}")
    
    results = run_2d_sensitivity(
        param1_name=param1,
        param2_name=param2,
        datasets=args.datasets,
        n_runs=args.n_runs,
        seed=args.seed
    )
    
    filename = f"sensitivity_2d_{param1}_{param2}.json"
    save_results(results, filename)
    
    print("\n" + "=" * 70)
    print("2D SENSITIVITY EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nTo generate 3D plots, run:")
    print(f"  python plotting/chart_sensitivity_3d.py --data {OUTPUT_DIR}/{filename}")


if __name__ == "__main__":
    main()
