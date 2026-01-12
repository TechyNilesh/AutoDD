"""
Generate Detector Selection Patterns Chart

Creates:
- charts/selection_pattern.png - Grouped bar chart

Usage:
    python plotting/chart_selection_pattern.py
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
rcParams['legend.fontsize'] = 11

# Data from supplementary table
detectors = ['ADWIN', 'DDM', 'EDDM', 'HDDM-A', 'HDDM-W', 'Page-H', 'KSWIN']
conditions = ['High Velocity', 'High Noise', 'High Dim.']

data = {
    'High Velocity': [28.4, 22.1, 8.3, 15.2, 12.6, 5.8, 7.6],
    'High Noise': [12.3, 8.5, 18.7, 10.4, 14.2, 8.1, 27.8],
    'High Dim.': [15.7, 11.2, 14.5, 12.8, 13.1, 7.2, 25.5],
}

x = np.arange(len(detectors))
width = 0.25

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(10, 5))

for i, condition in enumerate(conditions):
    offset = (i - 1) * width
    ax.bar(x + offset, data[condition], width, label=condition,
           color=colors[i], edgecolor='black', linewidth=1)

ax.set_ylabel('Selection Frequency (%)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(detectors, fontweight='bold')
ax.set_ylim(0, 35)

ax.legend(loc='upper right', framealpha=0.95, fontsize=11)

ax.tick_params(axis='both', which='major', labelsize=12)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_axisbelow(True)

plt.tight_layout()

output_path = os.path.join(CHARTS_DIR, "selection_pattern.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Chart saved: {output_path}")
