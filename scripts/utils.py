"""
Utility functions for AutoDD experiments.
Uses River's built-in metrics and stream utilities.
"""

import os
from river import stream, metrics


def iter_dataset(dataset_path, target='class'):
    """
    Iterate over a dataset using River's iter_csv.

    Args:
        dataset_path: Path to the CSV file
        target: Name of the target column

    Yields:
        (x, y): Tuple of feature dict and target value
    """
    return stream.iter_csv(dataset_path, target=target)


def get_metrics():
    """
    Get River metrics for prequential evaluation.

    Returns:
        dict of metric name -> metric object
    """
    return {
        'accuracy': metrics.Accuracy(),
        'f1': metrics.F1()
    }


def calculate_drift_f1(detected_drifts, true_drift_points, tolerance=100):
    """
    Calculate F1 score for drift detection.

    Args:
        detected_drifts: List of detected drift points
        true_drift_points: List of true drift points
        tolerance: Tolerance window for matching detections

    Returns:
        precision, recall, f1
    """
    if len(detected_drifts) == 0:
        return 0.0, 0.0, 0.0

    if len(true_drift_points) == 0:
        return 0.0, 0.0, 0.0

    tp = 0
    matched_true = set()

    for detected in detected_drifts:
        for i, true_point in enumerate(true_drift_points):
            if i not in matched_true and abs(detected - true_point) <= tolerance:
                tp += 1
                matched_true.add(i)
                break

    precision = tp / len(detected_drifts) if len(detected_drifts) > 0 else 0
    recall = tp / len(true_drift_points) if len(true_drift_points) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# Dataset names
SYNTHETIC_DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg']
REALWORLD_DATASETS = ['elec', 'weather', 'spam', 'airlines', 'poker']
ALL_DATASETS = SYNTHETIC_DATASETS + REALWORLD_DATASETS


def get_dataset_path(dataset_name, base_dir='datasets'):
    """Get the full path for a dataset."""
    return os.path.join(base_dir, f"{dataset_name}.csv")
