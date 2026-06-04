"""
Reset-strategy ablation for AutoDD.

Compares what the model does AFTER a drift is detected:
  - none : never reset the base classifier (upper bound on staleness)
  - hard : discard the model and relearn from scratch (current AutoDD behaviour)
  - warm : rebuild the model but replay the last `buffer_size` instances,
           so the fresh model is not blind right after the reset.

Usage:
    python scripts/reset_strategy_experiment.py --dataset elec --base_model gaussiannb
"""

import time
import os
import sys
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from river import metrics, stream
from AutoDD import AutoDD
from scripts.utils import get_base_model

import warnings
warnings.filterwarnings("ignore")


def run(dataset_name, reset_mode, base_model='gaussiannb', seed=42,
        buffer_size=200):
    model = get_base_model(base_model)
    detector = AutoDD(seed=seed)
    accuracy = metrics.Accuracy()
    f1 = metrics.MacroF1()
    recent = deque(maxlen=buffer_size)  # buffer of recent (x, y) for warm restart

    n = 0
    n_drifts = 0
    start = time.time()
    path = f"datasets/{dataset_name}.csv"

    for x, y in stream.iter_csv(path, target='class', converters={'class': str}):
        x = {k: float(v) for k, v in x.items()}
        y_pred = model.predict_one(x)
        if y_pred is not None:
            accuracy.update(y, y_pred)
            f1.update(y, y_pred)

        detector.update(x, y, y_pred if y_pred is not None else y)

        if detector.drift_detected:
            n_drifts += 1
            if reset_mode == 'hard':
                model = get_base_model(base_model)
            elif reset_mode == 'warm':
                model = get_base_model(base_model)
                for bx, by in recent:          # replay recent history
                    model.learn_one(bx, by)
            # reset_mode == 'none': keep the model unchanged

        model.learn_one(x, y)
        recent.append((x, y))
        n += 1

    return {
        'reset_mode': reset_mode,
        'accuracy': accuracy.get(),
        'f1': f1.get(),
        'n_drifts': n_drifts,
        'runtime': time.time() - start,
        'n': n,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--base_model', default='gaussiannb')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--buffer_size', type=int, default=200)
    args = p.parse_args()

    print(f"\nReset-strategy comparison on '{args.dataset}' (base={args.base_model}, "
          f"warm buffer={args.buffer_size})")
    print("=" * 68)
    print(f"{'reset_mode':<10} {'accuracy':>10} {'macro_f1':>10} "
          f"{'n_drifts':>10} {'runtime_s':>10}")
    print("-" * 68)
    for mode in ('none', 'hard', 'warm'):
        r = run(args.dataset, mode, args.base_model, args.seed, args.buffer_size)
        print(f"{r['reset_mode']:<10} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
              f"{r['n_drifts']:>10d} {r['runtime']:>10.2f}")
    print("=" * 68)
