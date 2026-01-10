"""
Generate Drift Count Tables

Creates:
- tables/drift_table.txt - Main drift count comparison
- tables/drift_<dataset>.txt - Per-dataset drift count tables

Usage:
    python plotting/table_drift_counts.py
"""

import os
import json
import pandas as pd
import numpy as np
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg', 'elec', 'weather', 'spam', 'airlines', 'poker']


def load_results():
    """Load all benchmark results into a DataFrame."""
    records = []
    for filepath in glob(os.path.join(RESULTS_DIR, "*.json")):
        if 'summary' in os.path.basename(filepath):
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            records.append({
                'model': data.get('model', ''),
                'dataset': data.get('dataset', ''),
                'n_drifts': data.get('n_drifts', 0),
                'run_count': data.get('run_count', 1)
            })
        except Exception as e:
            continue
    return pd.DataFrame(records)


def add_mean_row(df):
    """Add Mean row showing average across datasets for each model."""
    mean_values = {}
    for col in df.columns:
        values = []
        for val in df[col]:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue
        if values:
            mean_values[col] = np.mean(values)

    mean_row = pd.DataFrame([mean_values], index=['Mean'])
    return pd.concat([df, mean_row])


def generate_main_table(df_raw):
    """Generate main drift count table in pivot format."""
    pivot_df = df_raw.pivot_table(index='dataset', columns='model', values='n_drifts', aggfunc='mean')
    pivot_df.columns.name = None

    available_datasets = [d for d in DATASETS if d in pivot_df.index]
    pivot_df = pivot_df.reindex(available_datasets)

    pivot_df = add_mean_row(pivot_df)

    lines = []
    lines.append("=" * 140)
    lines.append("DRIFT COUNT COMPARISON TABLE")
    lines.append("=" * 140)
    lines.append("")

    models = list(pivot_df.columns)
    header = f"{'Model':<12s}"
    for model in models:
        header += f" {model:>12s}"
    lines.append(header)
    lines.append("-" * 140)

    for idx in pivot_df.index:
        row = f"{idx:<12s}"
        for model in models:
            val = pivot_df.loc[idx, model]
            if pd.isna(val):
                row += f" {'--':>12s}"
            else:
                row += f" {val:>12.1f}"
        lines.append(row)

    lines.append("-" * 140)

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Generating Drift Count Tables")
    print("=" * 60)

    df_raw = load_results()
    if df_raw.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df_raw)} records")

    table = generate_main_table(df_raw)
    path = os.path.join(TABLES_DIR, "drift_table.txt")
    with open(path, 'w') as f:
        f.write(table)
    print(f"\nMain table: {path}")
    print(table)

    print("\nDone!")


if __name__ == "__main__":
    main()
