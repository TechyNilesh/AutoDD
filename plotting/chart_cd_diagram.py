"""
Generate Critical Difference (CD) Diagram

Creates CD diagram comparing model rankings using Friedman test
and Nemenyi post-hoc test.

Creates:
- charts/cd_diagram.png - Critical difference diagram

Usage:
    python plotting/chart_cd_diagram.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from glob import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 18
plt.rcParams.update({'font.size': 18})

DATASETS = ['agr', 'airlines', 'elec', 'hyp', 'poker', 'rbf', 'rtg', 'sea', 'spam', 'weather']


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
            })
        except:
            continue
    return pd.DataFrame(records)


def add_mean_row(df):
    """Add mean row to dataframe."""
    mean_values = df.mean().to_frame().transpose()
    df_with_mean = pd.concat([df, mean_values], ignore_index=False)
    df_with_mean.rename({df_with_mean.index[-1]: 'Mean'}, inplace=True)
    return df_with_mean


def generate_cd_diagram():
    """Generate CD diagram."""
    try:
        from autorank import autorank, plot_stats
    except ImportError:
        print("Error: autorank package not installed.")
        print("Install with: pip install autorank")
        return None

    # Load data
    df_raw = load_results()
    if df_raw.empty:
        print("No results found!")
        return None

    # Create pivot table for accuracy
    temp_df = df_raw.pivot_table(
        index='dataset',
        columns='model',
        values='accuracy',
        aggfunc='mean'
    ).reset_index(drop=False).reset_index(drop=True)

    temp_df.columns.name = None
    temp_df = temp_df.set_index('dataset')

    # Reorder datasets
    temp_df = temp_df.loc[[d for d in DATASETS if d in temp_df.index]]

    # Add mean row
    temp_df = add_mean_row(temp_df)

    # Remove Mean row for autorank (it needs only data rows)
    data_df = temp_df.iloc[:-1]

    # Rename GaussianNB to "No Drift Detector" for consistent naming
    if 'GaussianNB' in data_df.columns:
        data_df = data_df.rename(columns={'GaussianNB': 'No Drift Detector'})

    print(f"Data shape: {data_df.shape}")
    print(f"Models: {list(data_df.columns)}")

    # Run autorank analysis
    res = autorank(
        data_df,
        alpha=0.05,
        order='descending',
        verbose=True,
        effect_size="cohen_d",
        force_mode='nonparametric'
    )

    # Plot CD diagram
    plot_stats(res, allow_insignificant=True)

    # Get current figure and customize
    fig = plt.gcf()
    ax = plt.gca()

    # Increase line thickness
    for line in ax.get_lines():
        line.set_linewidth(3.0)

    fig.set_size_inches(5, 5)

    output_path = os.path.join(CHARTS_DIR, "cd_diagram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Generating Critical Difference (CD) Diagram")
    print("=" * 60)

    path = generate_cd_diagram()
    if path:
        print(f"\nChart saved: {path}")
    else:
        print("Failed to generate CD diagram")

    print("\nDone!")


if __name__ == "__main__":
    main()
