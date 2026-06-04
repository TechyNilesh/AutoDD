"""
Standalone sweep of the proxy-reward drop_threshold parameter.

Does NOT modify any existing code or results. Patches _compute_reward
at runtime via subclassing so the original AutoDD is untouched.

Usage:
    python scripts/reward_threshold_sweep.py --dataset elec
"""

import time, os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from river import metrics, stream
from AutoDD import AutoDD
from scripts.utils import get_base_model

import warnings
warnings.filterwarnings("ignore")


class AutoDDThresh(AutoDD):
    """AutoDD with a configurable drop_threshold for the proxy reward."""

    def __init__(self, drop_threshold=0.05, **kwargs):
        super().__init__(**kwargs)
        self._drop_threshold = drop_threshold

    def _compute_reward(self):
        if len(self.accuracy_buffer) == 0 or self.prev_window_accuracy is None:
            return 0.0

        current_accuracy = np.mean(list(self.accuracy_buffer))
        accuracy_drop = self.prev_window_accuracy - current_accuracy
        significant_drop = accuracy_drop > self._drop_threshold

        if self._drift_detected:
            if significant_drop:
                reward = 0.5 + min(accuracy_drop * 2, 0.5)   # TP
            else:
                reward = -0.3                                  # FP
        else:
            if significant_drop:
                reward = -0.5 - min(accuracy_drop, 0.5)       # FN
            else:
                reward = 0.1                                   # TN

        return float(np.clip(reward, -1.0, 1.0))


def run_once(dataset_name, drop_threshold, base_model='gaussiannb', seed=42):
    model = get_base_model(base_model)
    detector = AutoDDThresh(drop_threshold=drop_threshold, seed=seed)
    acc = metrics.Accuracy()
    f1  = metrics.MacroF1()
    n_drifts = 0
    start = time.time()

    for x, y in stream.iter_csv(
            f"datasets/{dataset_name}.csv",
            target='class', converters={'class': str}):
        x = {k: float(v) for k, v in x.items()}
        y_pred = model.predict_one(x)
        if y_pred is not None:
            acc.update(y, y_pred)
            f1.update(y, y_pred)
        detector.update(x, y, y_pred if y_pred is not None else y)
        if detector.drift_detected:
            n_drifts += 1
            model = get_base_model(base_model)
        model.learn_one(x, y)

    return acc.get(), f1.get(), n_drifts, time.time() - start


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='elec')
    p.add_argument('--base_model', default='gaussiannb')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    print(f"\ndrop_threshold sweep on '{args.dataset}' (base={args.base_model}, seed={args.seed})")
    print(f"  current paper value: 0.05")
    print("=" * 68)
    print(f"{'threshold':>10} {'accuracy':>10} {'macro_f1':>10} {'n_drifts':>10} {'runtime_s':>10}")
    print("-" * 68)
    for t in thresholds:
        acc, f1, nd, rt = run_once(args.dataset, t, args.base_model, args.seed)
        marker = "  <-- paper" if t == 0.05 else ""
        print(f"{t:>10.2f} {acc:>10.4f} {f1:>10.4f} {nd:>10d} {rt:>10.2f}{marker}")
    print("=" * 68)
