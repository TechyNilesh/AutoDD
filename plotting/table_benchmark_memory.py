"""
Generate Benchmark Memory Tables

Creates:
- tables/memory_table.txt - Main memory comparison with mean and rank
- tables/memory_table_<base_model>.txt - Per base model memory tables

Usage:
    python plotting/table_benchmark_memory.py
    python plotting/table_benchmark_memory.py --base_model hat
    python plotting/table_benchmark_memory.py --base_model all
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
TABLES_DIR = os.path.join(SCRIPT_DIR, "tables")

os.makedirs(TABLES_DIR, exist_ok=True)

DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg', 'elec', 'weather', 'spam', 'airlines', 'poker']
BASE_MODELS = ['gaussiannb', 'hat']


def load_results(base_model=None):
    """Load all benchmark results into a DataFrame."""
    records = []
    for filepath in glob(os.path.join(RESULTS_DIR, "*.json")):
        if 'summary' in os.path.basename(filepath):
            continue
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            record_base_model = data.get('base_model', 'gaussiannb')

            if base_model and record_base_model != base_model:
                continue

            records.append({
                'model': data.get('model', ''),
                'detector': data.get('detector', ''),
                'dataset': data.get('dataset', ''),
                'total_memory_mb': data.get('total_memory_mb', 0),
                'base_model': record_base_model,
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


def add_rank_row(df):
    """Add Rank row showing mean rank for each model (lower memory = better = lower rank)."""
    data_df = df.iloc[:-1]

    rank_values = {}
    for col in data_df.columns:
        ranks = []
        for idx in data_df.index:
            val = data_df.loc[idx, col]
            try:
                ranks.append(float(val))
            except (ValueError, TypeError):
                ranks.append(np.nan)
        rank_values[col] = ranks

    rank_df = pd.DataFrame(rank_values, index=data_df.index)
    ranks_per_row = rank_df.rank(axis=1, ascending=True)
    mean_ranks = ranks_per_row.mean()

    rank_row = pd.DataFrame([{col: mean_ranks[col] for col in df.columns}], index=['Rank'])
    return pd.concat([df, rank_row])


def generate_main_table(df_raw, base_model_name=None):
    """Generate main memory table in pivot format."""
    pivot_df = df_raw.pivot_table(index='dataset', columns='model', values='total_memory_mb', aggfunc='mean')
    pivot_df.columns.name = None

    available_datasets = [d for d in DATASETS if d in pivot_df.index]
    pivot_df = pivot_df.reindex(available_datasets)

    pivot_df = add_mean_row(pivot_df)
    pivot_df = add_rank_row(pivot_df)

    lines = []
    lines.append("=" * 140)
    title = "MEMORY COMPARISON TABLE (MB)"
    if base_model_name:
        title += f" - {base_model_name.upper()}"
    lines.append(title)
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
            elif idx == 'Rank':
                row += f" {val:>12.1f}"
            else:
                row += f" {val:>12.2f}"
        lines.append(row)

    lines.append("-" * 140)
    lines.append("* Lower is better")

    return "\n".join(lines)


def main(base_model='all'):
    print("=" * 60)
    print("Generating Benchmark Memory Tables")
    print("=" * 60)

    if base_model == 'all':
        base_models_to_run = BASE_MODELS
    else:
        base_models_to_run = [base_model]

    for bm in base_models_to_run:
        print(f"\n--- Processing {bm.upper()} ---")

        df_raw = load_results(base_model=bm)
        if df_raw.empty:
            print(f"No results found for {bm}!")
            continue

        print(f"Loaded {len(df_raw)} records")

        table = generate_main_table(df_raw, base_model_name=bm)

        if len(base_models_to_run) > 1:
            filename = f"memory_table_{bm}.txt"
        else:
            filename = "memory_table.txt"

        path = os.path.join(TABLES_DIR, filename)
        with open(path, 'w') as f:
            f.write(table)
        print(f"Table saved: {path}")
        print(table)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Benchmark Memory Tables')
    parser.add_argument('--base_model', type=str, default='all',
                        choices=['gaussiannb', 'hat', 'all'],
                        help='Base classifier to use (default: all)')
    args = parser.parse_args()

    main(base_model=args.base_model)
