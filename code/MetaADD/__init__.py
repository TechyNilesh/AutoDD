"""
Meta-ADD: Active Drift Detection with Meta Learning

A meta-learning based drift detector that can classify different types
of concept drift using prototypical neural networks.

Based on the paper: "Automatic Learning to Detect Concept Drift" by Hang Yu et al.

Usage:
    # Pre-train the model (required before first use)
    from MetaADD import pretrain_meta_add, quick_pretrain
    model = quick_pretrain()  # For testing
    # OR
    model = pretrain_meta_add(n_streams_per_type=2000)  # Full training

    # Use pre-trained model
    from MetaADD import MetaADD, load_pretrained_model
    pretrained = load_pretrained_model()
    detector = MetaADD(pretrained_model=pretrained)
"""

from .meta_add import MetaADD
from .meta_feature_extractor import MetaFeatureExtractor
from .prototypical_network import PrototypicalNetwork
from .trainer import train_meta_detector
from .pretrain import (
    pretrain_meta_add,
    quick_pretrain,
    load_pretrained_model,
    generate_training_data
)

__all__ = [
    'MetaADD',
    'MetaFeatureExtractor',
    'PrototypicalNetwork',
    'train_meta_detector',
    'pretrain_meta_add',
    'quick_pretrain',
    'load_pretrained_model',
    'generate_training_data'
]

__version__ = '1.0.0'
