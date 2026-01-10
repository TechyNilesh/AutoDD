"""
Generate Benchmark Accuracy Tables

Creates:
- tables/accuracy_table.txt - Main accuracy comparison with mean and rank
- tables/accuracy_<dataset>.txt - Per-dataset accuracy tables

Usage:
    python plotting/table_benchmark_accuracy.py
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
SYNTHETIC = ['sea', 'hyp', 'agr', 'rbf', 'rtg']
REALWORLD = ['elec', 'weather', 'spam', 'airlines', 'poker']


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
                'accuracy': data.get('accuracy', 0),
                'run_count': data.get('run_count', 1)
            })
        except Exception as e:
            continue
    return pd.DataFrame(records)


def add_mean_row(df):
    """Add Mean row showing average across datasets for each model."""
    # Convert score strings to floats for mean calculation
    mean_values = {}
    for col in df.columns:
        values = []
        for val in df[col]:
            if isinstance(val, str) and '±' in val:
                values.append(float(val.split('±')[0]))
        if values:
            mean_values[col] = f"{np.mean(values):.3f}"

    mean_row = pd.DataFrame([mean_values], index=['Mean'])
    return pd.concat([df, mean_row])


def add_rank_row(df):
    """Add Rank row showing mean rank for each model (higher accuracy = better = lower rank)."""
    # Get data rows (exclude Mean row)
    data_df = df.iloc[:-1]

    # Calculate ranks per row (higher value = rank 1)
    rank_values = {}
    for col in data_df.columns:
        ranks = []
        for idx in data_df.index:
            val = data_df.loc[idx, col]
            if isinstance(val, str) and '±' in val:
                ranks.append(float(val.split('±')[0]))
            else:
                ranks.append(np.nan)
        rank_values[col] = ranks

    rank_df = pd.DataFrame(rank_values, index=data_df.index)
    # Rank: 1 = best (highest accuracy), ascending=False
    ranks_per_row = rank_df.rank(axis=1, ascending=False)
    mean_ranks = ranks_per_row.mean()

    rank_row = pd.DataFrame([{col: f"{mean_ranks[col]:.1f}" for col in df.columns}], index=['Rank'])
    return pd.concat([df, rank_row])


def generate_main_table(df_raw):
    """Generate main accuracy table in pivot format."""
    # Group by model and dataset, calculate mean and std
    agg_df = df_raw.groupby(['model', 'dataset'])['accuracy'].agg(['mean', 'std']).reset_index()

    # Create score column with mean±std format
    agg_df['score'] = [f"{mean:.3f}±{std:.3f}" for mean, std in zip(agg_df['mean'], agg_df['std'])]

    # Pivot: datasets as rows, models as columns
    pivot_df = agg_df.pivot_table(index='dataset', columns='model', values='score', aggfunc='first')
    pivot_df.columns.name = None

    # Reorder datasets
    available_datasets = [d for d in DATASETS if d in pivot_df.index]
    pivot_df = pivot_df.reindex(available_datasets)

    # Add Mean and Rank rows
    pivot_df = add_mean_row(pivot_df)
    pivot_df = add_rank_row(pivot_df)

    # Format as text table
    lines = []
    lines.append("=" * 140)
    lines.append("ACCURACY COMPARISON TABLE")
    lines.append("=" * 140)
    lines.append("")

    # Header row
    models = list(pivot_df.columns)
    header = f"{'Model':<12s}"
    for model in models:
        header += f" {model:>12s}"
    lines.append(header)
    lines.append("-" * 140)

    # Data rows
    for idx in pivot_df.index:
        row = f"{idx:<12s}"
        for model in models:
            val = pivot_df.loc[idx, model]
            if pd.isna(val):
                row += f" {'--':>12s}"
            else:
                row += f" {val:>12s}"
        lines.append(row)

    lines.append("-" * 140)

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Generating Benchmark Accuracy Tables")
    print("=" * 60)

    df_raw = load_results()
    if df_raw.empty:
        print("No results found!")
        return

    print(f"Loaded {len(df_raw)} records")

    # Main table
    table = generate_main_table(df_raw)
    path = os.path.join(TABLES_DIR, "accuracy_table.txt")
    with open(path, 'w') as f:
        f.write(table)
    print(f"\nMain table: {path}")
    print(table)

    print("\nDone!")


if __name__ == "__main__":
    main()
