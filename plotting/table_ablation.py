"""
Generate Ablation Study Table

Creates:
- tables/ablation_table.txt - Ablation study results

Usage:
    python plotting/table_ablation.py
"""

import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(CODE_DIR, "ablation_results", "ablation_summary.json")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)


def load_results():
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def generate_table(data):
    mean_acc = data['mean_accuracy']
    std_acc = data['std_accuracy']
    mean_f1 = data['mean_f1']
    configs = data['configurations']

    display_names = {
        'full': 'Full AutoDD',
        'no_temporal': 'w/o Temporal Features',
        'no_statistical': 'w/o Statistical Features',
        'no_stationarity': 'w/o Stationarity Features',
        'random': 'w/o LinUCB (Random)',
    }

    lines = []
    lines.append("=" * 70)
    lines.append("ABLATION STUDY RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Number of runs: {data['n_runs']}")
    lines.append(f"Datasets: {', '.join(data['datasets'])}")
    lines.append("")

    header = f"{'Configuration':<30s} {'Accuracy':>15s} {'F1':>10s} {'Delta':>10s}"
    lines.append(header)
    lines.append("-" * 70)

    full_acc = mean_acc.get('full', 0)

    for config in configs:
        name = display_names.get(config, config)
        acc = mean_acc.get(config, 0)
        std = std_acc.get(config, 0)
        f1 = mean_f1.get(config, 0)

        if config == 'full':
            delta_str = "--"
        else:
            delta = (acc - full_acc) * 100
            delta_str = f"{delta:+.1f}%"

        row = f"{name:<30s} {acc*100:>6.1f}%±{std*100:.1f}% {f1:>10.4f} {delta_str:>10s}"
        lines.append(row)

    lines.append("-" * 70)
    lines.append("")
    lines.append("Key Findings:")
    lines.append(f"- Full AutoDD: {full_acc*100:.1f}%")
    lines.append(f"- Removing Statistical: {(mean_acc.get('no_statistical', 0) - full_acc)*100:.1f}% drop")
    lines.append(f"- Random selection: {(mean_acc.get('random', 0) - full_acc)*100:.1f}% drop")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Generating Ablation Study Table")
    print("=" * 60)

    data = load_results()
    table = generate_table(data)

    path = os.path.join(TABLES_DIR, "ablation_table.txt")
    with open(path, 'w') as f:
        f.write(table)

    print(f"\nTable saved: {path}")
    print(table)
    print("\nDone!")


if __name__ == "__main__":
    main()
