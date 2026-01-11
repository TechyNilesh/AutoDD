"""
AutoDD: Automated Drift Detector Selection and Configuration

A **pluggable framework** for automatic drift detection in data streams.
Any River-compatible drift detector and classifier can be plugged in.

Key Features:
- Framework design: plug in any River-compatible detector or classifier
- Default portfolio: 7 diverse drift detectors (ADWIN, DDM, EDDM, etc.)
- Default classifier: GaussianNB (with HoeffdingTree, LogisticRegression available)
- Adaptive selection: LinUCB contextual bandit learns optimal detector
- Meta-features: 15 features capturing stream characteristics

Usage:
    # Default configuration (GaussianNB + all 7 detectors)
    >>> clf = AutoDDClassifier()

    # Use a different classifier
    >>> clf = AutoDDClassifier(base_classifier='HoeffdingTreeClassifier')

    # Plug in any River classifier
    >>> from river import ensemble
    >>> clf = AutoDDClassifier(base_classifier=ensemble.AdaBoostClassifier())

    # Custom detector/classifier portfolios
    >>> clf_portfolio = ClassifierPortfolio(classifiers=['GaussianNB', 'LogisticRegression'])
    >>> det_portfolio = DetectorPortfolio(detectors=['ADWIN', 'DDM'])
    >>> clf = AutoDDClassifier(classifier_portfolio=clf_portfolio, detector_portfolio=det_portfolio)
"""

from .autodd import AutoDD, AutoDDClassifier
from .meta_feature_extractor import MetaFeatureExtractor
from .detector_portfolio import DetectorPortfolio
from .classifier_portfolio import ClassifierPortfolio
from .linucb_agent import LinUCBAgent
from .hyperparameter_selector import CascadingHyperparameterSelector

__all__ = [
    'AutoDD',
    'AutoDDClassifier',
    'MetaFeatureExtractor',
    'DetectorPortfolio',
    'ClassifierPortfolio',
    'LinUCBAgent',
    'CascadingHyperparameterSelector'
]

__version__ = '1.0.0'
