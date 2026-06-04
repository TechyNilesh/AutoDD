"""
Generate REAL detector-selection timeline data by running AutoDD on a stream.

Runs actual AutoDD (LinUCB detector selection + Thompson-sampling HP tuning) on a
dataset with prequential evaluation, and records, in the schema consumed by
plotting/chart_detector_selection_combined.py:
  - n_instances
  - drift_points  : instances where AutoDD's active detector actually fired
  - timeline      : contiguous [start, end, detector] segments of the SELECTED detector
  - error_rate    : windowed prequential error (%) sampled at ~500 points

Unlike the previous hand-authored timeline_*.json, every value here is produced by
a genuine, reproducible single-seed run.

Usage:
    python plotting/generate_timeline_data.py --dataset hyp --n_instances 100000 --seed 42
"""

import os
import sys
import json
import argparse
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

from river import stream  # noqa: E402

from AutoDD import AutoDD  # noqa: E402
from scripts.utils import get_base_model  # noqa: E402

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(CODE_DIR, "benchmark_results")

# Hyperplane / SEA / etc. are continuous-drift synthetic streams; label for the JSON.
DRIFT_TYPE = {
    'hyp': 'incremental', 'sea': 'incremental', 'agr': 'sudden',
    'rbf': 'incremental', 'rtg': 'sudden',
}


def build_segments(per_instance_detector):
    """Compress a per-instance detector list into [start, end, detector] segments."""
    segments = []
    start = None
    current = None
    for i, det in enumerate(per_instance_detector):
        if det is None:
            continue
        if det != current:
            if current is not None:
                segments.append({'start': start, 'end': i, 'detector': current})
            current = det
            start = i
    if current is not None:
        segments.append({'start': start, 'end': len(per_instance_detector), 'detector': current})
    return segments


def block_majority_segments(per_instance_detector, n, block):
    """Summarise real selections as the dominant (modal) detector per `block` instances.

    This is a faithful summary of the genuine per-window selections - it does not
    invent data, it reports which real selection dominates each block - producing a
    legible band structure instead of a per-window barcode.
    """
    import collections
    blocks = []
    for b0 in range(0, n, block):
        seg = [d for d in per_instance_detector[b0:b0 + block] if d]
        if not seg:
            continue
        mode = collections.Counter(seg).most_common(1)[0][0]
        blocks.append((b0, min(b0 + block, n), mode))
    merged = []
    for st, en, det in blocks:
        if merged and merged[-1]['detector'] == det:
            merged[-1]['end'] = en
        else:
            merged.append({'start': st, 'end': en, 'detector': det})
    return merged


def generate(dataset, n_instances, seed, window_size, error_window, n_points,
             base_model='gaussiannb', smooth_block=0):
    dataset_path = os.path.join(CODE_DIR, "datasets", f"{dataset}.csv")
    print(f"Running REAL AutoDD on {dataset_path}")
    print(f"  n_instances={n_instances}, seed={seed}, window_size={window_size}, "
          f"error_window={error_window}")

    model = get_base_model(base_model)
    detector = AutoDD(window_size=window_size, alpha=1.0, lambda_reg=1.0,
                      n_hp_configs=5, seed=seed)

    per_instance_detector = [None] * n_instances
    drift_points = []

    # Rolling windowed error rate (deque of 0/1 errors over the last `error_window`)
    err_window = deque(maxlen=error_window)
    sample_stride = max(1, n_instances // n_points)
    error_x, error_y = [], []

    pos = 0
    for x, y in stream.iter_csv(dataset_path, target='class', converters={'class': str}):
        if pos >= n_instances:
            break
        x = {k: float(v) for k, v in x.items()}

        # Prequential: predict before learning
        y_pred = model.predict_one(x)
        err_window.append(0 if (y_pred is not None and y_pred == y) else 1)

        # Real AutoDD update (Level 1 + Level 2 selection happen at window boundaries)
        detector.update(x, y, y_pred if y_pred is not None else y)

        if detector.drift_detected:
            drift_points.append(pos)
            model = get_base_model(base_model)  # reset base model on drift (as in benchmark)

        model.learn_one(x, y)

        per_instance_detector[pos] = detector.current_detector

        # Sample the windowed error rate
        if (pos + 1) % sample_stride == 0:
            error_x.append(pos + 1)
            error_y.append(100.0 * sum(err_window) / len(err_window))

        pos += 1
        if pos % 10000 == 0:
            print(f"  ...{pos}/{n_instances}  drifts={len(drift_points)}  "
                  f"current={detector.current_detector}")

    raw_segments = build_segments(per_instance_detector[:pos])
    if smooth_block and smooth_block > 0:
        timeline = block_majority_segments(per_instance_detector[:pos], pos, smooth_block)
    else:
        timeline = raw_segments

    data = {
        'dataset': dataset,
        'n_instances': pos,
        'drift_points': drift_points,
        'drift_type': DRIFT_TYPE.get(dataset, 'incremental'),
        'timeline': timeline,
        'error_rate': {'x': error_x, 'y': error_y},
        'provenance': {
            'generated_by': 'plotting/generate_timeline_data.py',
            'real_run': True,
            'seed': seed,
            'window_size': window_size,
            'error_window': error_window,
            'base_model': base_model,
            'smooth_block': smooth_block,
            'raw_segment_count': len(raw_segments),
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"timeline_{dataset}.json")
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDetected drifts: {len(drift_points)} -> {drift_points[:10]}"
          f"{' ...' if len(drift_points) > 10 else ''}")
    print(f"Timeline segments ({len(timeline)}):")
    for s in timeline:
        print(f"  {s['start']:>6}-{s['end']:>6}  ({s['end']-s['start']:>5})  {s['detector']}")
    print(f"\nSaved real timeline -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate REAL AutoDD detector-selection timeline")
    p.add_argument('--dataset', type=str, default='hyp')
    p.add_argument('--n_instances', type=int, default=100000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window_size', type=int, default=100)
    p.add_argument('--error_window', type=int, default=500,
                   help='Sliding window (instances) for the local error-rate curve')
    p.add_argument('--n_points', type=int, default=500,
                   help='Number of points to sample on the error-rate curve')
    p.add_argument('--base_model', type=str, default='gaussiannb', choices=['gaussiannb', 'hat'])
    p.add_argument('--smooth_block', type=int, default=0,
                   help='If >0, report the dominant (modal) real selection per this many '
                        'instances instead of per-window (e.g. 2000) for a legible figure')
    args = p.parse_args()
    generate(args.dataset, args.n_instances, args.seed, args.window_size,
             args.error_window, args.n_points, args.base_model, args.smooth_block)
