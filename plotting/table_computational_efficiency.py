"""
Generate Computational Efficiency Table

Combines results across ALL runs and generates a plain text table
showing Time (ms/inst) and Memory (MB) for all detectors.

Creates:
- tables/computational_efficiency.txt - Plain text table

Usage:
    python plotting/table_computational_efficiency.py
"""

import os
import json
import numpy as np
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

# Model display names (for nice formatting)
MODEL_NAMES = {
    'AUTODD': 'AutoDD',
    'MetaADD': 'Meta-ADD',
    'PUDD': 'PUDD',
    'ADDM': 'ADDM',
    'MDDM': 'MDDM',
    'GaussianNB': 'No Drift Detector',
    'ADWIN': 'ADWIN',
    'DDM': 'DDM',
    'EDDM': 'EDDM',
    'HDDM_A': 'HDDM-A',
    'HDDM_W': 'HDDM-W',
    'PAGEHINKLEY': 'Page-Hinkley',
    'KSWIN': 'KSWIN',
}


def load_all_results():
    """Load ALL benchmark results and combine across runs."""
    records = []
    for filepath in glob(os.path.join(RESULTS_DIR, "*.json")):
        if 'summary' in os.path.basename(filepath):
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            total_runtime = data.get('total_runtime', 0)
            n_instances = data.get('n_instances', 1)
            ms_per_instance = (total_runtime / n_instances) * 1000 if n_instances > 0 else 0
            records.append({
                'model': data.get('model', ''),
                'dataset': data.get('dataset', ''),
                'run_id': data.get('run_id', 1),
                'ms_per_inst': ms_per_instance,
                'memory_mb': data.get('total_memory_mb', 0),
            })
        except:
            continue
    return records


def compute_combined_stats(records):
    """Compute mean ± std time and memory across ALL runs for each model."""
    model_stats = {}

    for rec in records:
        model = rec['model']
        if not model:
            continue
        # Skip GaussianNB (No Drift Detector) - it's a baseline without drift detection
        if model == 'GaussianNB':
            continue
        if model not in model_stats:
            model_stats[model] = {'time': [], 'memory': []}
        model_stats[model]['time'].append(rec['ms_per_inst'])
        model_stats[model]['memory'].append(rec['memory_mb'])

    combined_stats = {}
    for model, stats in model_stats.items():
        if stats['time'] and stats['memory']:
            combined_stats[model] = {
                'time_mean': np.mean(stats['time']),
                'time_std': np.std(stats['time']),
                'memory_mean': np.mean(stats['memory']),
                'memory_std': np.std(stats['memory']),
                'n_samples': len(stats['time'])
            }

    return combined_stats


def generate_table(combined_stats):
    """Generate plain text table sorted by time."""

    # Sort by time (fastest first)
    sorted_models = sorted(combined_stats.items(), key=lambda x: x[1]['time_mean'])

    lines = []
    lines.append("=" * 55)
    lines.append("Table: Computational Efficiency")
    lines.append("=" * 55)
    lines.append("")
    lines.append(f"{'Method':<20s} {'Time (ms/inst)':>15s} {'Memory (MB)':>15s}")
    lines.append("-" * 55)

    for model, stats in sorted_models:
        display_name = MODEL_NAMES.get(model, model)
        time_val = stats['time_mean']
        memory_val = stats['memory_mean']

        # Format with 2 decimal places for time, 1 for memory
        lines.append(f"{display_name:<20s} {time_val:>15.2f} {memory_val:>15.1f}")

    lines.append("-" * 55)
    lines.append("")
    lines.append(f"* Results combined across {sorted_models[0][1]['n_samples']} measurements per method")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Generating Computational Efficiency Table")
    print("(Combined across all runs)")
    print("=" * 60)

    records = load_all_results()
    if not records:
        print("No results found!")
        return

    print(f"Loaded {len(records)} total records")

    combined_stats = compute_combined_stats(records)
    print(f"Models: {len(combined_stats)}")

    # Generate table
    table = generate_table(combined_stats)

    # Save to file
    output_path = os.path.join(TABLES_DIR, "computational_efficiency.txt")
    with open(output_path, 'w') as f:
        f.write(table)
    print(f"\nSaved to: {output_path}")

    # Print table
    print("\n" + table)

    print("\nDone!")


if __name__ == "__main__":
    main()
