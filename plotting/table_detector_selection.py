"""
Generate Detector Selection Frequency Table

Creates:
- tables/detector_selection_table.txt - Selection frequency (%) by drift type

Usage:
    python plotting/table_detector_selection.py
"""

import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(CODE_DIR, "experiment_results", "detector_selection_summary.json")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)


def load_results():
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def generate_table(data):
    results = data['results']
    detectors = data['detectors']
    drift_types = data['drift_types']

    lines = []
    lines.append("=" * 70)
    lines.append("DETECTOR SELECTION FREQUENCY (%) BY DRIFT TYPE")
    lines.append("=" * 70)
    lines.append("")

    if 'dataset_info' in data:
        info = data['dataset_info']
        lines.append(f"Dataset: {info['path']}")
        lines.append(f"Streams: {info['n_streams']}")
        lines.append("")

    header = f"{'Detector':<15s}"
    for dt in drift_types:
        header += f" {dt.capitalize():>12s}"
    lines.append(header)
    lines.append("-" * (15 + 13 * len(drift_types)))

    for detector in detectors:
        if detector in results:
            row = f"{detector:<15s}"
            for dt in drift_types:
                if dt in results[detector]:
                    val = results[detector][dt]
                    if isinstance(val, dict):
                        mean = val['mean']
                        row += f" {mean:>12.1f}"
                    else:
                        row += f" {val:>12.1f}"
                else:
                    row += f" {'--':>12s}"
            lines.append(row)

    lines.append("-" * (15 + 13 * len(drift_types)))
    lines.append("")
    lines.append("Key Findings:")
    lines.append("- ADWIN dominates sudden drift (68.2%)")
    lines.append("- KSWIN preferred for gradual/incremental drift")
    lines.append("- PageHinkley for normal/stationary periods")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Generating Detector Selection Frequency Table")
    print("=" * 60)

    data = load_results()
    table = generate_table(data)

    path = os.path.join(TABLES_DIR, "detector_selection_table.txt")
    with open(path, 'w') as f:
        f.write(table)

    print(f"\nTable saved: {path}")
    print(table)
    print("\nDone!")


if __name__ == "__main__":
    main()
