"""
Detector Portfolio for AutoDD

AutoDD is a framework - any River-compatible drift detector can be plugged in.

Default portfolio includes K=7 diverse detectors spanning multiple paradigms:
- Statistical: ADWIN, Page-Hinkley
- Error-based: DDM, EDDM, HDDM-A, HDDM-W
- Distribution-based: KSWIN

Users can:
1. Use default detectors with default/custom hyperparameters
2. Plug in any River-compatible drift detector
3. Create custom portfolios for experimentation

Supports hyperparameter configuration ranges for adaptive selection.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from river import drift
from river.drift import binary as binary_drift
import numpy as np


# Hyperparameter configuration ranges for each detector
# Format: {param_name: (min, max, default, scale)}
# scale: 'linear', 'log', or 'int'
DETECTOR_PARAM_RANGES = {
    'ADWIN': {
        'delta': (0.0001, 0.1, 0.002, 'log'),  # Confidence for detecting change
    },
    'DDM': {
        'warm_start': (10, 100, 30, 'int'),  # Minimum instances before detection
        'warning_threshold': (1.5, 3.0, 2.0, 'linear'),  # Warning level threshold
        'drift_threshold': (2.0, 4.0, 3.0, 'linear'),  # Drift level threshold
    },
    'EDDM': {
        'warm_start': (10, 100, 30, 'int'),  # Minimum instances before detection
        'alpha': (0.90, 0.99, 0.95, 'linear'),  # Warning threshold
        'beta': (0.85, 0.95, 0.9, 'linear'),  # Drift threshold
    },
    'HDDM_A': {
        'drift_confidence': (0.0001, 0.01, 0.001, 'log'),  # Confidence for drift
        'warning_confidence': (0.001, 0.05, 0.005, 'log'),  # Confidence for warning
    },
    'HDDM_W': {
        'drift_confidence': (0.0001, 0.01, 0.001, 'log'),  # Confidence for drift
        'warning_confidence': (0.001, 0.05, 0.005, 'log'),  # Confidence for warning
        'lambda_val': (0.01, 0.1, 0.05, 'linear'),  # EWMA smoothing factor
    },
    'PageHinkley': {
        'min_instances': (10, 100, 30, 'int'),  # Minimum instances before detection
        'delta': (0.001, 0.01, 0.005, 'log'),  # Magnitude threshold
        'threshold': (10, 100, 50, 'linear'),  # Cumulative sum threshold
        'alpha': (0.99, 0.9999, 0.9999, 'linear'),  # Forgetting factor
    },
    'KSWIN': {
        'alpha': (0.001, 0.05, 0.005, 'log'),  # Significance level for KS test
        'window_size': (50, 500, 100, 'int'),  # Size of sliding window
        'stat_size': (10, 100, 30, 'int'),  # Size of statistic window
    },
}


class DetectorPortfolio:
    """
    Manages a portfolio of drift detectors.

    AutoDD is a framework that works with any River-compatible drift detector.
    The default portfolio contains K=7 diverse detectors from different paradigms,
    but users can plug in custom detectors.

    Parameters
    ----------
    detectors : list of str, optional
        List of detector names to include. If None, uses all 7 default detectors.
    custom_detectors : dict, optional
        Dictionary mapping names to custom detector configs:
        {'MyDetector': {'class': MyClass, 'params': {...}, 'paradigm': 'custom'}}

    Attributes
    ----------
    detectors : dict
        Dictionary mapping detector names to detector instances.
    detector_names : list
        List of detector names.
    n_detectors : int
        Number of detectors in portfolio (K).

    Examples
    --------
    >>> # Use default portfolio
    >>> portfolio = DetectorPortfolio()
    >>> print(portfolio.detector_names)
    ['ADWIN', 'DDM', 'EDDM', 'HDDM_A', 'HDDM_W', 'PageHinkley', 'KSWIN']

    >>> # Use only specific detectors
    >>> portfolio = DetectorPortfolio(detectors=['ADWIN', 'DDM'])

    >>> # Add custom River detector
    >>> from river.drift import DummyDriftDetector  # example
    >>> custom = {'Custom': {'class': DummyDriftDetector, 'params': {}, 'paradigm': 'custom'}}
    >>> portfolio = DetectorPortfolio(custom_detectors=custom)
    """

    # Default detector configurations from the paper
    # Note: DDM, EDDM, HDDM_A, HDDM_W are in river.drift.binary
    DEFAULT_CONFIGS = {
        'ADWIN': {
            'class': drift.ADWIN,
            'params': {'delta': 0.002},
            'paradigm': 'statistical'
        },
        'DDM': {
            'class': binary_drift.DDM,
            'params': {
                'warm_start': 30,
                'warning_threshold': 2.0,
                'drift_threshold': 3.0
            },
            'paradigm': 'error_based'
        },
        'EDDM': {
            'class': binary_drift.EDDM,
            'params': {
                'warm_start': 30,
                'alpha': 0.95,
                'beta': 0.9
            },
            'paradigm': 'error_based'
        },
        'HDDM_A': {
            'class': binary_drift.HDDM_A,
            'params': {
                'drift_confidence': 0.001,
                'warning_confidence': 0.005
            },
            'paradigm': 'error_based'
        },
        'HDDM_W': {
            'class': binary_drift.HDDM_W,
            'params': {
                'drift_confidence': 0.001,
                'warning_confidence': 0.005,
                'lambda_val': 0.05
            },
            'paradigm': 'error_based'
        },
        'PageHinkley': {
            'class': drift.PageHinkley,
            'params': {
                'min_instances': 30,
                'delta': 0.005,
                'threshold': 50,
                'alpha': 0.9999
            },
            'paradigm': 'statistical'
        },
        'KSWIN': {
            'class': drift.KSWIN,
            'params': {
                'alpha': 0.005,
                'window_size': 100,
                'stat_size': 30
            },
            'paradigm': 'distribution_based'
        }
    }

    def __init__(
        self,
        detectors: Optional[List[str]] = None,
        custom_detectors: Optional[Dict[str, Dict]] = None,
        param_ranges: Optional[Dict[str, Dict[str, Tuple]]] = None
    ):
        """
        Initialize the detector portfolio.

        Parameters
        ----------
        detectors : list of str, optional
            Detector names to include. If None, uses all 7 default detectors.
        custom_detectors : dict, optional
            Custom detector configurations to add:
            {'MyDetector': {'class': MyClass, 'params': {...}, 'paradigm': 'custom'}}
        param_ranges : dict, optional
            Custom hyperparameter search ranges. Overrides defaults.
            Format: {detector_name: {param_name: (min, max, default, scale)}}
            Scale can be: 'linear', 'log', or 'int'

            Example:
            {
                'ADWIN': {'delta': (0.0001, 0.5, 0.002, 'log')},
                'DDM': {'drift_threshold': (2.0, 5.0, 3.0, 'linear')}
            }

        Examples
        --------
        >>> # Use default HP ranges
        >>> portfolio = DetectorPortfolio()

        >>> # Custom HP ranges for specific detectors
        >>> custom_ranges = {
        ...     'ADWIN': {'delta': (0.001, 0.1, 0.01, 'log')},
        ...     'DDM': {'drift_threshold': (2.5, 4.0, 3.0, 'linear')}
        ... }
        >>> portfolio = DetectorPortfolio(param_ranges=custom_ranges)
        """
        # Merge default and custom configs
        self.all_configs = self.DEFAULT_CONFIGS.copy()
        if custom_detectors:
            for name, config in custom_detectors.items():
                if 'class' not in config:
                    raise ValueError(f"Custom detector '{name}' must have 'class' key")
                # Ensure required keys exist
                if 'params' not in config:
                    config['params'] = {}
                if 'paradigm' not in config:
                    config['paradigm'] = 'custom'
                self.all_configs[name] = config

        # Merge default and custom param ranges
        self._param_ranges = DETECTOR_PARAM_RANGES.copy()
        if param_ranges:
            for detector_name, ranges in param_ranges.items():
                if detector_name not in self._param_ranges:
                    self._param_ranges[detector_name] = {}
                self._param_ranges[detector_name].update(ranges)

        # Select which detectors to include
        if detectors is None:
            detectors = list(self.DEFAULT_CONFIGS.keys())  # Only default detectors by default

        self.detector_names = detectors
        self.n_detectors = len(detectors)

        # Initialize detectors
        self.detectors: Dict[str, Any] = {}
        self.configs: Dict[str, Dict] = {}

        for name in detectors:
            if name not in self.all_configs:
                raise ValueError(f"Unknown detector: {name}. "
                                f"Available: {list(self.all_configs.keys())}")

            config = self.all_configs[name]
            self.configs[name] = config
            self.detectors[name] = self._create_detector(name)

    def _create_detector(self, name: str, params: Optional[Dict] = None) -> Any:
        """Create a new detector instance."""
        config = self.all_configs[name]
        if params is None:
            params = config['params']
        return config['class'](**params)

    def get_detector(self, name: str) -> Any:
        """
        Get a detector by name.

        Parameters
        ----------
        name : str
            Detector name.

        Returns
        -------
        detector : drift detector
            The detector instance.
        """
        if name not in self.detectors:
            raise ValueError(f"Detector {name} not in portfolio.")
        return self.detectors[name]

    def get_detector_by_index(self, index: int) -> Tuple[str, Any]:
        """
        Get a detector by index.

        Parameters
        ----------
        index : int
            Detector index (0 to K-1).

        Returns
        -------
        name : str
            Detector name.
        detector : drift detector
            The detector instance.
        """
        name = self.detector_names[index]
        return name, self.detectors[name]

    def update_detector(self, name: str, value: float) -> bool:
        """
        Update a detector with a new value and check for drift.

        Parameters
        ----------
        name : str
            Detector name.
        value : float
            Value to feed to detector (typically error: 0 or 1).

        Returns
        -------
        drift_detected : bool
            Whether drift was detected.
        """
        detector = self.detectors[name]
        detector.update(value)
        return detector.drift_detected

    def reset_detector(self, name: str):
        """
        Reset a detector to initial state.

        Parameters
        ----------
        name : str
            Detector name.
        """
        # Create a new instance to reset
        self.detectors[name] = self._create_detector(name)

    def reset_all(self):
        """Reset all detectors to initial state."""
        for name in self.detector_names:
            self.reset_detector(name)

    def get_paradigm(self, name: str) -> str:
        """
        Get the paradigm of a detector.

        Parameters
        ----------
        name : str
            Detector name.

        Returns
        -------
        paradigm : str
            One of 'statistical', 'error_based', 'distribution_based'.
        """
        return self.configs[name]['paradigm']

    def get_all_paradigms(self) -> Dict[str, str]:
        """Return dictionary mapping detector names to paradigms."""
        return {name: self.get_paradigm(name) for name in self.detector_names}

    def get_detector_info(self, name: str) -> Dict:
        """
        Get information about a detector.

        Parameters
        ----------
        name : str
            Detector name.

        Returns
        -------
        info : dict
            Dictionary with 'name', 'paradigm', 'params'.
        """
        config = self.configs[name]
        return {
            'name': name,
            'paradigm': config['paradigm'],
            'params': config['params']
        }

    def list_detectors(self) -> List[Dict]:
        """Return list of all detector information."""
        return [self.get_detector_info(name) for name in self.detector_names]

    def __len__(self) -> int:
        """Return number of detectors."""
        return self.n_detectors

    def __iter__(self):
        """Iterate over detector names."""
        return iter(self.detector_names)

    def __contains__(self, name: str) -> bool:
        """Check if detector is in portfolio."""
        return name in self.detector_names

    # ==================== Hyperparameter Selection Methods ====================

    def get_param_ranges(self, detector_name: str) -> Dict[str, Tuple]:
        """
        Get hyperparameter ranges for a detector.

        Parameters
        ----------
        detector_name : str
            Name of the detector.

        Returns
        -------
        ranges : dict
            Dictionary mapping param names to (min, max, default, scale) tuples.
        """
        if detector_name not in self._param_ranges:
            return {}  # Custom detectors may not have ranges defined
        return self._param_ranges[detector_name]

    def set_param_ranges(self, detector_name: str, ranges: Dict[str, Tuple]):
        """
        Set custom hyperparameter ranges for a detector.

        Parameters
        ----------
        detector_name : str
            Name of the detector.
        ranges : dict
            Dictionary mapping param names to (min, max, default, scale) tuples.
            Example: {'delta': (0.001, 0.1, 0.01, 'log')}
        """
        if detector_name not in self._param_ranges:
            self._param_ranges[detector_name] = {}
        self._param_ranges[detector_name].update(ranges)

    @staticmethod
    def sample_param(param_range: Tuple, rng: Optional[np.random.Generator] = None) -> Union[int, float]:
        """
        Sample a parameter value from its range.

        Parameters
        ----------
        param_range : tuple
            (min, max, default, scale) tuple.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        value : int or float
            Sampled parameter value.
        """
        if rng is None:
            rng = np.random.default_rng()

        min_val, max_val, default, scale = param_range

        if scale == 'int':
            return int(rng.integers(int(min_val), int(max_val) + 1))
        elif scale == 'log':
            log_min, log_max = np.log(min_val), np.log(max_val)
            return float(np.exp(rng.uniform(log_min, log_max)))
        else:  # linear
            return float(rng.uniform(min_val, max_val))

    def sample_hyperparameters(self, detector_name: str,
                                rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Sample random hyperparameters for a detector.

        Parameters
        ----------
        detector_name : str
            Name of the detector.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        params : dict
            Dictionary of sampled hyperparameters.
        """
        if rng is None:
            rng = np.random.default_rng()

        ranges = self.get_param_ranges(detector_name)
        params = {}

        for param_name, param_range in ranges.items():
            params[param_name] = self.sample_param(param_range, rng)

        # Several river detectors impose ordering constraints between paired
        # parameters that independent sampling can violate. Enforce them here so
        # randomly generated configurations are always valid.

        # KSWIN: window_size >= 2 * stat_size, else "Sample larger than population".
        if 'window_size' in params and 'stat_size' in params:
            max_stat = params['window_size'] // 2
            if params['stat_size'] > max_stat:
                params['stat_size'] = int(max(1, max_stat))

        # EDDM: alpha (warning) must be >= beta (drift).
        if 'alpha' in params and 'beta' in params:
            if params['alpha'] < params['beta']:
                params['alpha'], params['beta'] = params['beta'], params['alpha']

        # DDM: drift_threshold must be >= warning_threshold.
        if 'drift_threshold' in params and 'warning_threshold' in params:
            if params['drift_threshold'] < params['warning_threshold']:
                params['drift_threshold'], params['warning_threshold'] = (
                    params['warning_threshold'], params['drift_threshold'])

        # HDDM: warning_confidence must be >= drift_confidence.
        if 'warning_confidence' in params and 'drift_confidence' in params:
            if params['warning_confidence'] < params['drift_confidence']:
                params['warning_confidence'], params['drift_confidence'] = (
                    params['drift_confidence'], params['warning_confidence'])

        return params

    def create_detector_with_params(self, detector_name: str,
                                     params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a detector with specified hyperparameters.

        Parameters
        ----------
        detector_name : str
            Name of the detector.
        params : dict, optional
            Hyperparameters. If None, uses defaults.

        Returns
        -------
        detector : drift detector
            New detector instance with specified parameters.
        """
        if detector_name not in self.all_configs:
            raise ValueError(f"Unknown detector: {detector_name}")

        config = self.all_configs[detector_name]

        if params is None:
            params = config['params']

        return config['class'](**params)

    def get_detector_configurations(self, detector_name: str,
                                     n_configs: int = 5,
                                     include_default: bool = True,
                                     rng: Optional[np.random.Generator] = None) -> List[Dict[str, Any]]:
        """
        Generate multiple hyperparameter configurations for a detector.

        Parameters
        ----------
        detector_name : str
            Name of the detector.
        n_configs : int
            Number of configurations to generate.
        include_default : bool
            Whether to include default configuration.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        configs : list of dict
            List of hyperparameter configurations.
        """
        if rng is None:
            rng = np.random.default_rng()

        configs = []

        # Add default configuration first if requested
        if include_default:
            default_config = self.DEFAULT_CONFIGS[detector_name]['params'].copy()
            configs.append(default_config)
            n_configs -= 1

        # Generate random configurations
        for _ in range(n_configs):
            configs.append(self.sample_hyperparameters(detector_name, rng))

        return configs

    def create_augmented_portfolio(self, n_variants: int = 3,
                                    rng: Optional[np.random.Generator] = None) -> Dict[str, List[Any]]:
        """
        Create an augmented portfolio with multiple hyperparameter variants per detector.

        Parameters
        ----------
        n_variants : int
            Number of variants per detector.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        augmented : dict
            Dictionary mapping detector names to lists of detector instances.
        """
        if rng is None:
            rng = np.random.default_rng()

        augmented = {}

        for name in self.detector_names:
            configs = self.get_detector_configurations(name, n_variants, include_default=True, rng=rng)
            augmented[name] = [self.create_detector_with_params(name, cfg) for cfg in configs]

        return augmented

    def get_all_param_ranges(self) -> Dict[str, Dict[str, Tuple]]:
        """
        Get all hyperparameter ranges for all detectors.

        Returns
        -------
        all_ranges : dict
            Dictionary mapping detector names to their parameter ranges.
        """
        return self._param_ranges.copy()

    @staticmethod
    def get_default_param_ranges() -> Dict[str, Dict[str, Tuple]]:
        """
        Get default hyperparameter ranges (class-level defaults).

        Returns
        -------
        default_ranges : dict
            Dictionary mapping detector names to their default parameter ranges.
        """
        return DETECTOR_PARAM_RANGES.copy()

    # ==================== Convenience Methods ====================

    @classmethod
    def from_river_detector(cls, detector: Any, name: Optional[str] = None) -> 'DetectorPortfolio':
        """
        Create a portfolio from a single River detector.

        This is the simplest way to use AutoDD with a custom detector.

        Parameters
        ----------
        detector : river drift detector
            Any River-compatible drift detector instance.
        name : str, optional
            Name for the detector. Defaults to class name.

        Returns
        -------
        portfolio : DetectorPortfolio
            Portfolio containing just this detector.

        Examples
        --------
        >>> from river.drift import ADWIN
        >>> detector = ADWIN(delta=0.001)
        >>> portfolio = DetectorPortfolio.from_river_detector(detector, 'ADWIN_sensitive')
        """
        if name is None:
            name = type(detector).__name__

        custom = {
            name: {
                'class': type(detector),
                'params': {},  # User's detector is already configured
                'paradigm': 'custom'
            }
        }

        return cls(detectors=[name], custom_detectors=custom)


class DetectorWrapper:
    """
    Wrapper for a drift detector that tracks statistics.

    Parameters
    ----------
    name : str
        Detector name.
    detector : drift detector
        The underlying detector.

    Attributes
    ----------
    n_updates : int
        Number of updates.
    n_drifts : int
        Number of drifts detected.
    drift_points : list
        List of instance numbers where drifts were detected.
    """

    def __init__(self, name: str, detector: Any):
        self.name = name
        self.detector = detector
        self.n_updates = 0
        self.n_drifts = 0
        self.drift_points: List[int] = []

    def update(self, value: float) -> bool:
        """
        Update the detector and track statistics.

        Parameters
        ----------
        value : float
            Value to feed to detector.

        Returns
        -------
        drift_detected : bool
            Whether drift was detected.
        """
        self.n_updates += 1
        self.detector.update(value)

        if self.detector.drift_detected:
            self.n_drifts += 1
            self.drift_points.append(self.n_updates)
            return True

        return False

    @property
    def drift_detected(self) -> bool:
        """Check if drift was detected in last update."""
        return self.detector.drift_detected

    def reset(self):
        """Reset detector and statistics."""
        self.detector = type(self.detector)(**self.detector.__dict__)
        self.n_updates = 0
        self.n_drifts = 0
        self.drift_points = []

    def get_stats(self) -> Dict:
        """Return detector statistics."""
        return {
            'name': self.name,
            'n_updates': self.n_updates,
            'n_drifts': self.n_drifts,
            'drift_points': self.drift_points
        }
