"""
Vector recreation of the concept figure 'type_of_drift_detection' (Fig. 1):
the four canonical concept-drift patterns as 2-class scatter over time.

Produces a crisp vector PDF (+ PNG) so the figure stays high quality when zoomed.

    python plotting/chart_drift_types.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import to_rgb

rcParams['font.family'] = 'serif'
rcParams['font.weight'] = 'bold'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

BLUE = '#2a4bd7'
RED = '#d81e0a'
N = 280
S = 9          # marker size
YLO, YHI = 0.30, 0.70
SPREAD = 0.05


def _band(ax, x, y0, color):
    ax.scatter(x, y0 + SPREAD * np.random.randn(len(x)), s=S, c=color,
               alpha=0.85, edgecolors='none', zorder=3)


def main():
    np.random.seed(7)
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 5.2), sharex=True)
    titles = ['Sudden\nDrift', 'Gradual\nDrift', 'Incremental\nDrift', 'Recurring\nContext']
    t = np.sort(np.random.rand(N))

    # 1. Sudden: abrupt switch at t = 0.5
    ax = axes[0]
    m = t < 0.5
    _band(ax, t[m], YLO, BLUE)
    _band(ax, t[~m], YHI, RED)

    # 2. Gradual: B appears with rising probability through a transition window
    ax = axes[1]
    pB = np.clip((t - 0.40) / 0.30, 0, 1)
    isB = np.random.rand(N) < pB
    _band(ax, t[~isB], YLO, BLUE)
    _band(ax, t[isB], YHI, RED)

    # 3. Incremental: smooth continuous shift, colour blends blue -> red
    ax = axes[2]
    y = YLO + (YHI - YLO) * t
    cb, cr = np.array(to_rgb(BLUE)), np.array(to_rgb(RED))
    cols = [tuple((1 - tt) * cb + tt * cr) for tt in t]
    ax.scatter(t, y + 0.04 * np.random.randn(N), s=S, c=cols,
               alpha=0.85, edgecolors='none', zorder=3)

    # 4. Recurring: A -> B -> A (old concept returns)
    ax = axes[3]
    mB = (t < 0.34) | (t > 0.66)
    _band(ax, t[mB], YLO, BLUE)
    _band(ax, t[~mB], YHI, RED)

    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Data\nDistribution\n$P(x,y)$', fontsize=7,
                      rotation=0, ha='right', va='center', labelpad=2)
        ax.text(-0.30, 0.5, title, transform=ax.transAxes, fontsize=12,
                ha='center', va='center', fontweight='bold')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

    # "Time ->" arrow under the bottom panel
    axes[-1].annotate('', xy=(1.0, -0.18), xytext=(0.0, -0.18),
                      xycoords='axes fraction',
                      arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))
    axes[-1].text(0.5, -0.34, 'Time', transform=axes[-1].transAxes,
                  ha='center', va='top', fontsize=11, fontweight='bold')

    plt.tight_layout(h_pad=0.6)
    pdf = os.path.join(CHARTS_DIR, 'type_of_drift_detection.pdf')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(pdf[:-4] + '.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('saved', pdf)


if __name__ == '__main__':
    main()
