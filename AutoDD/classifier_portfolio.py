"""
Classifier Portfolio for AutoDD

Manages a portfolio of base classifiers for streaming learning.
AutoDD is a framework - any River-compatible classifier can be plugged in.

Default portfolio includes:
- Naive Bayes: GaussianNB (default)
- Tree-based: HoeffdingTreeClassifier (VFDT)
- Linear: LogisticRegression, Perceptron

Note: Unlike detectors, classifiers don't have hyperparameter search.
Users specify classifier params directly when creating the portfolio.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# River classifiers
from river import naive_bayes, tree, linear_model


class ClassifierPortfolio:
    """
    Manages a portfolio of base classifiers for AutoDD.

    AutoDD is a framework that works with any River-compatible classifier.
    This portfolio provides default classifiers but allows full customization.

    Note: Classifiers use fixed hyperparameters (no HP search like detectors).
    Specify custom params via classifier_params or custom_classifiers.

    Parameters
    ----------
    classifiers : list of str, optional
        List of classifier names to include. If None, uses all defaults.
    classifier_params : dict, optional
        Custom parameters for default classifiers.
        Example: {'HoeffdingTreeClassifier': {'grace_period': 100}}
    custom_classifiers : dict, optional
        Dictionary mapping names to custom classifier configs:
        {'MyClassifier': {'class': MyClass, 'params': {...}}}

    Attributes
    ----------
    classifiers : dict
        Dictionary mapping classifier names to classifier instances.
    classifier_names : list
        List of classifier names.
    n_classifiers : int
        Number of classifiers in portfolio.
    default_classifier : str
        Name of the default classifier (GaussianNB).

    Examples
    --------
    >>> # Use default portfolio
    >>> portfolio = ClassifierPortfolio()
    >>> clf = portfolio.get_classifier('GaussianNB')

    >>> # Use only specific classifiers
    >>> portfolio = ClassifierPortfolio(classifiers=['GaussianNB', 'HoeffdingTreeClassifier'])

    >>> # Custom params for default classifiers
    >>> portfolio = ClassifierPortfolio(
    ...     classifier_params={'HoeffdingTreeClassifier': {'grace_period': 100, 'delta': 1e-5}}
    ... )

    >>> # Add custom River classifier
    >>> from river import ensemble
    >>> custom = {'AdaBoost': {'class': ensemble.AdaBoostClassifier, 'params': {}}}
    >>> portfolio = ClassifierPortfolio(custom_classifiers=custom)
    """

    # Default classifier configurations
    DEFAULT_CONFIGS = {
        'GaussianNB': {
            'class': naive_bayes.GaussianNB,
            'params': {},
            'type': 'naive_bayes',
            'description': 'Gaussian Naive Bayes - fast, simple, default choice'
        },
        'HoeffdingTreeClassifier': {
            'class': tree.HoeffdingTreeClassifier,
            'params': {
                'grace_period': 200,
                'delta': 1e-7,
                'tau': 0.05
            },
            'type': 'tree',
            'description': 'Hoeffding Tree (VFDT) - tree-based streaming learner'
        },
        'LogisticRegression': {
            'class': linear_model.LogisticRegression,
            'params': {},
            'type': 'linear',
            'description': 'Logistic Regression - linear model for streaming'
        },
        'Perceptron': {
            'class': linear_model.Perceptron,
            'params': {},
            'type': 'linear',
            'description': 'Perceptron - simple linear classifier'
        },
    }

    # Default classifier to use when none specified
    DEFAULT_CLASSIFIER = 'GaussianNB'

    def __init__(
        self,
        classifiers: Optional[List[str]] = None,
        classifier_params: Optional[Dict[str, Dict]] = None,
        custom_classifiers: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize the classifier portfolio.

        Parameters
        ----------
        classifiers : list of str, optional
            Classifier names to include. If None, uses all defaults.
        classifier_params : dict, optional
            Custom parameters for default classifiers. Merges with defaults.
            Example: {'HoeffdingTreeClassifier': {'grace_period': 100}}
        custom_classifiers : dict, optional
            Custom classifier configurations to add.
        """
        # Start with default configs (deep copy)
        self.all_configs = {}
        for name, config in self.DEFAULT_CONFIGS.items():
            self.all_configs[name] = {
                'class': config['class'],
                'params': config['params'].copy(),
                'type': config['type'],
                'description': config['description']
            }

        # Apply custom params to default classifiers
        if classifier_params:
            for name, params in classifier_params.items():
                if name in self.all_configs:
                    self.all_configs[name]['params'].update(params)

        # Add custom classifiers
        if custom_classifiers:
            for name, config in custom_classifiers.items():
                if 'class' not in config:
                    raise ValueError(f"Custom classifier '{name}' must have 'class' key")
                # Ensure required keys exist
                if 'params' not in config:
                    config['params'] = {}
                if 'type' not in config:
                    config['type'] = 'custom'
                if 'description' not in config:
                    config['description'] = f'Custom classifier: {name}'
                self.all_configs[name] = config

        # Select which classifiers to include
        if classifiers is None:
            classifiers = list(self.DEFAULT_CONFIGS.keys())

        self.classifier_names = classifiers
        self.n_classifiers = len(classifiers)
        self.default_classifier = self.DEFAULT_CLASSIFIER

        # Validate and initialize classifiers
        self.classifiers: Dict[str, Any] = {}
        self.configs: Dict[str, Dict] = {}

        for name in classifiers:
            if name not in self.all_configs:
                raise ValueError(
                    f"Unknown classifier: {name}. "
                    f"Available: {list(self.all_configs.keys())}"
                )
            config = self.all_configs[name]
            self.configs[name] = config
            self.classifiers[name] = self._create_classifier(name)

    def _create_classifier(self, name: str, params: Optional[Dict] = None) -> Any:
        """Create a new classifier instance."""
        config = self.all_configs[name]
        if params is None:
            params = config['params']
        return config['class'](**params)

    def get_classifier(self, name: Optional[str] = None) -> Any:
        """
        Get a classifier by name.

        Parameters
        ----------
        name : str, optional
            Classifier name. If None, returns default (GaussianNB).

        Returns
        -------
        classifier : river classifier
            The classifier instance.
        """
        if name is None:
            name = self.default_classifier
        if name not in self.classifiers:
            raise ValueError(f"Classifier {name} not in portfolio.")
        return self.classifiers[name]

    def get_classifier_by_index(self, index: int) -> Tuple[str, Any]:
        """
        Get a classifier by index.

        Parameters
        ----------
        index : int
            Classifier index.

        Returns
        -------
        name : str
            Classifier name.
        classifier : river classifier
            The classifier instance.
        """
        name = self.classifier_names[index]
        return name, self.classifiers[name]

    def create_classifier(self, name: Optional[str] = None, params: Optional[Dict] = None) -> Any:
        """
        Create a new classifier instance (for reset after drift).

        Parameters
        ----------
        name : str, optional
            Classifier name. If None, uses default.
        params : dict, optional
            Custom parameters. If None, uses configured defaults.

        Returns
        -------
        classifier : river classifier
            New classifier instance.
        """
        if name is None:
            name = self.default_classifier
        return self._create_classifier(name, params)

    def reset_classifier(self, name: str):
        """
        Reset a classifier to initial state.

        Parameters
        ----------
        name : str
            Classifier name.
        """
        self.classifiers[name] = self._create_classifier(name)

    def reset_all(self):
        """Reset all classifiers to initial state."""
        for name in self.classifier_names:
            self.reset_classifier(name)

    def get_classifier_type(self, name: str) -> str:
        """
        Get the type of a classifier.

        Parameters
        ----------
        name : str
            Classifier name.

        Returns
        -------
        type : str
            One of 'naive_bayes', 'tree', 'linear', 'custom'.
        """
        return self.configs[name]['type']

    def get_classifier_params(self, name: str) -> Dict:
        """
        Get the parameters of a classifier.

        Parameters
        ----------
        name : str
            Classifier name.

        Returns
        -------
        params : dict
            Classifier parameters.
        """
        return self.configs[name]['params'].copy()

    def set_classifier_params(self, name: str, params: Dict):
        """
        Update parameters for a classifier.

        Parameters
        ----------
        name : str
            Classifier name.
        params : dict
            Parameters to update.
        """
        if name not in self.all_configs:
            raise ValueError(f"Unknown classifier: {name}")
        self.all_configs[name]['params'].update(params)
        self.configs[name]['params'].update(params)
        # Recreate classifier with new params
        self.classifiers[name] = self._create_classifier(name)

    def get_classifier_info(self, name: str) -> Dict:
        """
        Get information about a classifier.

        Parameters
        ----------
        name : str
            Classifier name.

        Returns
        -------
        info : dict
            Dictionary with 'name', 'type', 'params', 'description'.
        """
        config = self.configs[name]
        return {
            'name': name,
            'type': config['type'],
            'params': config['params'].copy(),
            'description': config['description']
        }

    def list_classifiers(self) -> List[Dict]:
        """Return list of all classifier information."""
        return [self.get_classifier_info(name) for name in self.classifier_names]

    def __len__(self) -> int:
        """Return number of classifiers."""
        return self.n_classifiers

    def __iter__(self):
        """Iterate over classifier names."""
        return iter(self.classifier_names)

    def __contains__(self, name: str) -> bool:
        """Check if classifier is in portfolio."""
        return name in self.classifier_names

    # ==================== Convenience Methods ====================

    @classmethod
    def from_river_classifier(cls, classifier: Any, name: Optional[str] = None) -> 'ClassifierPortfolio':
        """
        Create a portfolio from a single River classifier.

        This is the simplest way to use AutoDD with a custom classifier.

        Parameters
        ----------
        classifier : river classifier
            Any River-compatible classifier instance.
        name : str, optional
            Name for the classifier. Defaults to class name.

        Returns
        -------
        portfolio : ClassifierPortfolio
            Portfolio containing just this classifier.

        Examples
        --------
        >>> from river import ensemble
        >>> clf = ensemble.AdaBoostClassifier()
        >>> portfolio = ClassifierPortfolio.from_river_classifier(clf)
        """
        if name is None:
            name = type(classifier).__name__

        custom = {
            name: {
                'class': type(classifier),
                'params': {},  # User's classifier is already configured
                'type': 'custom',
                'description': f'Custom: {name}'
            }
        }

        return cls(classifiers=[name], custom_classifiers=custom)

    @classmethod
    def default_only(cls) -> 'ClassifierPortfolio':
        """
        Create a portfolio with only the default classifier (GaussianNB).

        Returns
        -------
        portfolio : ClassifierPortfolio
            Portfolio with only GaussianNB.
        """
        return cls(classifiers=['GaussianNB'])
