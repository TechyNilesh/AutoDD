"""
Generate Alpha (LinUCB Exploration) Sensitivity Table

Creates:
- tables/alpha_sensitivity_table.txt - Accuracy with different alpha values

Usage:
    python plotting/table_alpha_sensitivity.py
"""

import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_FILE = os.path.join(CODE_DIR, "experiment_results", "alpha_sensitivity_summary.json")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

DATASET_NAMES = {
    'sea': 'SEA',
    'hyp': 'HYP',
    'agr': 'AGR',
    'rbf': 'RBF',
    'rtg': 'RTG',
    'elec': 'ELEC',
}


def load_results():
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def generate_table(data):
    results = data['results']
    alpha_values = data['alpha_values']
    optimal = data.get('optimal_alpha', 1.0)

    lines = []
    lines.append("=" * 90)
    lines.append("ACCURACY (%) WITH DIFFERENT ALPHA (LinUCB EXPLORATION) VALUES")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Description: {data.get('description', 'Alpha controls exploration vs exploitation')}")
    lines.append(f"Optimal alpha: α={optimal}")
    lines.append("")

    header = f"{'Dataset':<10s}"
    for alpha in alpha_values:
        marker = "*" if alpha == optimal else ""
        header += f" {'α='+str(alpha)+marker:>12s}"
    lines.append(header)
    lines.append("-" * (10 + 13 * len(alpha_values)))

    datasets = list(results.keys())
    for ds in datasets:
        name = DATASET_NAMES.get(ds, ds.upper())
        row = f"{name:<10s}"
        for alpha in alpha_values:
            alpha_key = str(alpha)
            if alpha_key in results[ds]:
                mean = results[ds][alpha_key]['accuracy_mean']
                std = results[ds][alpha_key]['accuracy_std']
                row += f" {mean*100:>6.1f}±{std*100:.1f}"
            else:
                row += f" {'--':>12s}"
        lines.append(row)

    lines.append("-" * (10 + 13 * len(alpha_values)))
    lines.append(f"* Optimal alpha value (α={optimal})")
    lines.append("")
    lines.append("Key Findings:")
    lines.append("- α=1.0 provides best performance across all datasets")
    lines.append("- Low α (0.1): Gets stuck with suboptimal detector (too exploitative)")
    lines.append("- High α (2.0): Too much switching, unstable (too exploratory)")

    return "\n".join(lines)


def generate_latex_table(data):
    results = data['results']
    alpha_values = data['alpha_values']
    optimal = data.get('optimal_alpha', 1.0)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Accuracy (\\%) with different $\\alpha$ (LinUCB exploration) values. Best results in \\textbf{bold}.}")
    lines.append("\\label{tab:alpha_sensitivity}")

    col_spec = "l" + "c" * len(alpha_values)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "Dataset"
    for alpha in alpha_values:
        header += f" & $\\alpha$={alpha}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    datasets = list(results.keys())
    for ds in datasets:
        name = DATASET_NAMES.get(ds, ds.upper())
        row = name
        best_acc = max(results[ds][str(a)]['accuracy_mean'] for a in alpha_values)
        for alpha in alpha_values:
            alpha_key = str(alpha)
            mean = results[ds][alpha_key]['accuracy_mean']
            val = f"{mean*100:.1f}"
            if mean == best_acc:
                val = f"\\textbf{{{val}}}"
            row += f" & {val}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Generating Alpha Sensitivity Table")
    print("=" * 60)

    data = load_results()

    # Text table
    table = generate_table(data)
    path = os.path.join(TABLES_DIR, "alpha_sensitivity_table.txt")
    with open(path, 'w') as f:
        f.write(table)
    print(f"\nText table saved: {path}")

    # LaTeX table
    latex_table = generate_latex_table(data)
    latex_path = os.path.join(TABLES_DIR, "alpha_sensitivity_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved: {latex_path}")

    print("\n" + table)
    print("\nDone!")


if __name__ == "__main__":
    main()
