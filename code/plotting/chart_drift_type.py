"""
Generate Per-Drift-Type Performance Chart

Creates:
- charts/drift_type_performance.png - Grouped bar chart

Usage:
    python plotting/chart_drift_type.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 10

# Data from supplementary table
drift_types = ['Sudden', 'Gradual', 'Incremental', 'Recurring']
methods = ['ADWIN', 'DDM', 'EDDM', 'HDDM-A', 'HDDM-W', 'Page-H', 'KSWIN',
           'MDDM', 'ADDM', 'PUDD', 'Meta-ADD', 'AutoDD']

data = {
    'Sudden': [82.4, 79.8, 75.2, 81.7, 80.1, 81.9, 76.5, 81.3, 82.9, 82.1, 84.6, 88.2],
    'Gradual': [71.3, 74.6, 78.4, 72.1, 76.2, 70.8, 77.8, 75.8, 77.2, 79.5, 82.3, 81.7],
    'Incremental': [68.5, 72.1, 74.8, 69.8, 73.5, 68.2, 75.2, 74.1, 75.6, 76.8, 78.9, 80.4],
    'Recurring': [75.2, 71.8, 69.3, 74.5, 74.6, 75.8, 72.1, 73.9, 74.8, 75.4, 77.8, 79.6],
}

x = np.arange(len(drift_types))
width = 0.07
colors = ['#7aa6c2'] * 7 + ['#f9a26c'] * 4 + ['#d62728']

fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    values = [data[dt][i] for dt in drift_types]
    offset = (i - 5.5) * width
    bars = ax.bar(x + offset, values, width, label=method, color=colors[i],
                   edgecolor='black', linewidth=0.8)

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(drift_types, fontweight='bold')
ax.set_ylim(65, 92)

ax.legend(loc='upper right', ncol=3, framealpha=0.95, fontsize=9)

ax.tick_params(axis='both', which='major', labelsize=12)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_axisbelow(True)

plt.tight_layout()

output_path = os.path.join(CHARTS_DIR, "drift_type_performance.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Chart saved: {output_path}")
