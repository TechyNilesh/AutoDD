"""
Utility functions for AutoDD experiments.
Uses River's built-in metrics and stream utilities.
"""

import os
from river import stream, metrics, naive_bayes, tree


# Base model configurations
BASE_MODELS = {
    'gaussiannb': {
        'name': 'GaussianNB',
        'class': naive_bayes.GaussianNB,
    },
    'hat': {
        'name': 'HAT',
        'class': tree.HoeffdingAdaptiveTreeClassifier,
    }
}


def get_base_model(base_model_name='gaussiannb'):
    """
    Get a new instance of the base model.

    Args:
        base_model_name: 'gaussiannb' or 'hat'

    Returns:
        A new model instance
    """
    config = BASE_MODELS.get(base_model_name, BASE_MODELS['gaussiannb'])
    return config['class']()


def get_base_model_display_name(base_model_name='gaussiannb'):
    """Get the display name for the base model."""
    config = BASE_MODELS.get(base_model_name, BASE_MODELS['gaussiannb'])
    return config['name']


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


# Dataset names
SYNTHETIC_DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg']
REALWORLD_DATASETS = ['elec', 'weather', 'spam', 'airlines', 'poker']
ALL_DATASETS = SYNTHETIC_DATASETS + REALWORLD_DATASETS


def get_dataset_path(dataset_name, base_dir='datasets'):
    """Get the full path for a dataset."""
    return os.path.join(base_dir, f"{dataset_name}.csv")
