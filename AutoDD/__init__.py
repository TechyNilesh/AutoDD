"""
AutoDD: Automated Drift Detector Selection and Configuration

A framework that automatically selects and configures drift detectors
for data streams using contextual multi-armed bandits (LinUCB).

Based on the paper: "AutoDD: Automated Drift Detector Selection and
Configuration for Data Streams"

Key Components:
- Meta-Feature Extraction: 15 features capturing stream characteristics
- Detector Portfolio: 7 diverse drift detectors
- LinUCB Agent: Contextual bandit for adaptive selection
"""

from .autodd import AutoDD, AutoDDClassifier
from .meta_feature_extractor import MetaFeatureExtractor
from .detector_portfolio import DetectorPortfolio
from .linucb_agent import LinUCBAgent
from .hyperparameter_selector import CascadingHyperparameterSelector

__all__ = [
    'AutoDD',
    'AutoDDClassifier',
    'MetaFeatureExtractor',
    'DetectorPortfolio',
    'LinUCBAgent',
    'CascadingHyperparameterSelector'
]

__version__ = '1.0.0'
