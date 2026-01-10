"""
AutoDD: Automated Drift Detector Selection and Configuration

Main class that orchestrates the automatic selection and configuration
of drift detectors using contextual multi-armed bandits (LinUCB).

Algorithm:
1. Extract meta-features from stream window
2. Select detector using LinUCB based on meta-features
3. Apply selected detector to check for drift
4. Compute reward based on accuracy drop
5. Update LinUCB with observed reward
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from .meta_feature_extractor import MetaFeatureExtractor
from .detector_portfolio import DetectorPortfolio
from .linucb_agent import LinUCBAgent
from .hyperparameter_selector import CascadingHyperparameterSelector


class AutoDD:
    """
    AutoDD: Automated Drift Detector Selection and Configuration.

    Automatically selects the optimal drift detector by learning the
    relationship between stream characteristics (meta-features) and
    detector performance using LinUCB contextual bandits.

    Parameters
    ----------
    window_size : int, default=100
        Size of sliding window for meta-feature extraction.
    alpha : float, default=1.0
        LinUCB exploration parameter. Higher values = more exploration.
    detectors : list of str, optional
        Detector names to include in portfolio. If None, uses all 7 detectors.

    Attributes
    ----------
    drift_detected : bool
        Whether drift was detected in the last window.
    current_detector : str
        Name of the currently selected detector.
    n_detections : int
        Total number of drifts detected.
    selection_history : list
        History of detector selections.

    Examples
    --------
    >>> from AutoDD import AutoDD
    >>> autodd = AutoDD(window_size=100, alpha=1.0)
    >>> for x, y_true in stream:
    ...     y_pred = model.predict_one(x)
    ...     autodd.update(x, y_true, y_pred)
    ...     if autodd.drift_detected:
    ...         print(f"Drift detected by {autodd.current_detector}")
    ...         model = reset_model()
    ...     model.learn_one(x, y_true)

    References
    ----------
    AutoDD Paper: "AutoDD: Automated Drift Detector Selection and
    Configuration for Data Streams"
    """

    def __init__(
        self,
        window_size: int = 100,  # Optimal window size based on experiments
        alpha: float = 1.0,
        lambda_reg: float = 1.0,  # Regularization parameter for LinUCB
        detectors: Optional[List[str]] = None,
        cascading_hp: bool = True,  # Enable cascading hyperparameter selection
        n_hp_configs: int = 5,  # Number of HP configurations per detector
        seed: Optional[int] = None
    ):
        self.window_size = window_size
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.cascading_hp = cascading_hp
        self.n_hp_configs = n_hp_configs

        # Initialize components
        self.meta_extractor = MetaFeatureExtractor(window_size=window_size)
        self.portfolio = DetectorPortfolio(detectors=detectors)

        # Cascading Hyperparameter Selector (Level 2: Thompson Sampling)
        if cascading_hp:
            self.hp_selector = CascadingHyperparameterSelector(
                portfolio=self.portfolio,
                n_configs_per_detector=n_hp_configs,
                include_default=True,
                seed=seed
            )
        else:
            self.hp_selector = None

        # LinUCB agent (K arms, 15 features) - Level 1: Detector Selection
        self.n_arms = len(self.portfolio)
        self.n_features = 15  # Fixed: 15 meta-features from paper
        self.agent = LinUCBAgent(
            n_arms=self.n_arms,
            n_features=self.n_features,
            alpha=alpha,
            lambda_reg=lambda_reg
        )

        # Map detector names to arm indices
        self.detector_to_arm = {
            name: i for i, name in enumerate(self.portfolio.detector_names)
        }
        self.arm_to_detector = {
            i: name for name, i in self.detector_to_arm.items()
        }

        # State tracking
        self._drift_detected = False
        self.current_detector: Optional[str] = None
        self.current_arm: Optional[int] = None
        self._n_detections = 0

        # Accuracy tracking for reward computation
        self.accuracy_buffer: deque = deque(maxlen=window_size)
        self.prev_window_accuracy: Optional[float] = None

        # History
        self.selection_history: List[str] = []
        self.drift_history: List[Tuple[int, str]] = []
        self.reward_history: List[float] = []

        # Instance counter
        self._n_instances = 0
        self._window_count = 0

        # Active detector for continuous updates (Fix 2)
        self._active_detector_name: Optional[str] = None
        self._active_detector = None
        self._last_meta_features: Optional[np.ndarray] = None
        
        # Cascading HP tracking
        self._current_hp_config_idx: Optional[int] = None
        self._current_hp_params: Optional[Dict] = None

    def update(self, x: Dict, y_true, y_pred) -> 'AutoDD':
        """
        Update AutoDD with a new instance.

        Parameters
        ----------
        x : dict
            Feature dictionary.
        y_true : any
            True label.
        y_pred : any
            Predicted label.

        Returns
        -------
        self : AutoDD
        """
        self._n_instances += 1
        self._drift_detected = False

        # Track accuracy for reward computation
        correct = 1.0 if y_true == y_pred else 0.0
        self.accuracy_buffer.append(correct)

        # Calculate error for detector
        error = 0.0 if y_true == y_pred else 1.0

        # Update meta-feature extractor
        self.meta_extractor.update(x, y_true, y_pred)

        # Continuous detector update (Fix 2)
        # Update active detector with each instance
        if self._active_detector is not None:
            self._active_detector.update(error)
            if self._active_detector.drift_detected:
                self._drift_detected = True
                self._n_detections += 1
                self.drift_history.append((self._n_instances, self._active_detector_name))

                # Compute reward for drift detection
                if self._last_meta_features is not None:
                    reward = self._compute_reward()
                    self.reward_history.append(reward)
                    self.agent.update(self.current_arm, self._last_meta_features, reward)

                # Reset detector after drift
                self.portfolio.reset_detector(self._active_detector_name)
                self._active_detector = self.portfolio.get_detector(self._active_detector_name)

        # Select/reselect detector at window boundaries
        if self._n_instances % self.window_size == 0:
            self._process_window()

        return self

    def _process_window(self):
        """
        Process a complete window: cascading selection of detector and hyperparameters.
        
        Level 1 (LinUCB): Select detector based on stream meta-features
        Level 2 (Thompson Sampling): Select hyperparameters for chosen detector
        """
        self._window_count += 1

        # Get meta-features
        if not self.meta_extractor.is_ready():
            # Select first detector with default config if no meta-features yet
            if self._active_detector is None:
                first_detector = self.portfolio.detector_names[0]
                self._active_detector_name = first_detector
                if self.cascading_hp and self.hp_selector:
                    # Use Thompson Sampling for initial HP selection
                    self._current_hp_config_idx, self._current_hp_params = \
                        self.hp_selector.select_config(first_detector)
                    self._active_detector = self.hp_selector.create_detector(
                        first_detector, self._current_hp_params
                    )
                else:
                    self._active_detector = self.portfolio.get_detector(first_detector)
                self.current_detector = first_detector
                self.current_arm = 0
            return

        meta_features = self.meta_extractor.get_features()

        # If we have a previous selection, update both levels with reward
        if self._last_meta_features is not None and self.current_arm is not None:
            reward = self._compute_reward()
            self.reward_history.append(reward)
            
            # Level 1: Update LinUCB (detector selection)
            self.agent.update(self.current_arm, self._last_meta_features, reward)
            
            # Level 2: Update Thompson Sampling (hyperparameter selection)
            if self.cascading_hp and self.hp_selector and self._current_hp_config_idx is not None:
                self.hp_selector.update(
                    self._active_detector_name,
                    self._current_hp_config_idx,
                    reward
                )

        # Level 1: Select detector using LinUCB
        selected_arm = self.agent.select_arm(meta_features)
        selected_detector = self.arm_to_detector[selected_arm]

        self.current_arm = selected_arm
        self.current_detector = selected_detector
        self.selection_history.append(selected_detector)

        # Level 2: Select hyperparameters using Thompson Sampling
        self._active_detector_name = selected_detector
        if self.cascading_hp and self.hp_selector:
            self._current_hp_config_idx, self._current_hp_params = \
                self.hp_selector.select_config(selected_detector)
            self._active_detector = self.hp_selector.create_detector(
                selected_detector, self._current_hp_params
            )
        else:
            self.portfolio.reset_detector(selected_detector)
            self._active_detector = self.portfolio.get_detector(selected_detector)
            self._current_hp_config_idx = None
            self._current_hp_params = None

        # Save meta-features for next reward computation
        self._last_meta_features = meta_features.copy()

        # Update previous accuracy for next reward computation
        self.prev_window_accuracy = np.mean(list(self.accuracy_buffer))

    def _compute_reward(self) -> float:
        """
        Compute reward based on detection quality (Fix 3).

        Improved reward function that better captures:
        - True positive: drift detected AND accuracy dropped -> high positive reward
        - False positive: drift detected but no accuracy drop -> negative reward
        - False negative: no detection but accuracy dropped -> negative reward
        - True negative: no detection and stable accuracy -> small positive reward

        Returns
        -------
        reward : float
            Reward value in [-1, 1].
        """
        if len(self.accuracy_buffer) == 0:
            return 0.0

        current_accuracy = np.mean(list(self.accuracy_buffer))

        if self.prev_window_accuracy is None:
            return 0.0

        # Accuracy drop (positive = performance degradation)
        accuracy_drop = self.prev_window_accuracy - current_accuracy

        # Threshold for significant accuracy drop
        drop_threshold = 0.05

        # Check if there was significant performance degradation
        significant_drop = accuracy_drop > drop_threshold

        if self._drift_detected:
            if significant_drop:
                # True Positive: Correctly detected drift
                # Reward proportional to accuracy drop (good detection)
                reward = 0.5 + min(accuracy_drop * 2, 0.5)
            else:
                # False Positive: Detected drift but performance was stable
                # Penalize unnecessary model resets
                reward = -0.3
        else:
            if significant_drop:
                # False Negative: Missed drift, performance degraded
                # Penalize proportional to how much accuracy dropped
                reward = -0.5 - min(accuracy_drop, 0.5)
            else:
                # True Negative: Correctly did not detect drift
                # Small positive reward for stable operation
                reward = 0.1

        return float(np.clip(reward, -1.0, 1.0))

    @property
    def drift_detected(self) -> bool:
        """Whether drift was detected in the last window."""
        return self._drift_detected

    @property
    def n_detections(self) -> int:
        """Total number of drifts detected."""
        return self._n_detections

    def get_selection_distribution(self) -> Dict[str, float]:
        """
        Get the distribution of detector selections.

        Returns
        -------
        distribution : dict
            Dictionary mapping detector names to selection frequencies.
        """
        if not self.selection_history:
            return {name: 0.0 for name in self.portfolio.detector_names}

        counts = {}
        for name in self.portfolio.detector_names:
            counts[name] = self.selection_history.count(name) / len(self.selection_history)

        return counts

    def get_drift_info(self) -> Dict:
        """
        Get information about drift detection.

        Returns
        -------
        info : dict
            Dictionary with drift detection information.
        """
        return {
            'drift_detected': self._drift_detected,
            'current_detector': self.current_detector,
            'n_detections': self._n_detections,
            'n_instances': self._n_instances,
            'n_windows': self._window_count,
            'selection_distribution': self.get_selection_distribution()
        }

    def get_agent_stats(self) -> Dict:
        """Get LinUCB agent statistics."""
        return self.agent.get_arm_stats()

    def get_meta_features(self) -> np.ndarray:
        """Get current meta-feature vector."""
        if self.meta_extractor.current_features is None:
            return np.zeros(self.n_features)
        return self.meta_extractor.current_features

    def get_meta_feature_dict(self) -> Dict[str, float]:
        """Get current meta-features as dictionary."""
        return self.meta_extractor.get_feature_dict()

    def predict_best_detector(self, meta_features: Optional[np.ndarray] = None) -> str:
        """
        Predict the best detector for given meta-features.

        Parameters
        ----------
        meta_features : numpy.ndarray, optional
            Meta-feature vector. If None, uses current features.

        Returns
        -------
        detector : str
            Name of predicted best detector.
        """
        if meta_features is None:
            meta_features = self.get_meta_features()

        arm = self.agent.select_arm(meta_features)
        return self.arm_to_detector[arm]

    def get_ucb_values(self, meta_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get UCB values for all detectors.

        Parameters
        ----------
        meta_features : numpy.ndarray, optional
            Meta-feature vector. If None, uses current features.

        Returns
        -------
        ucb_values : dict
            Dictionary mapping detector names to UCB values.
        """
        if meta_features is None:
            meta_features = self.get_meta_features()

        ucb = self.agent.get_ucb_values(meta_features)
        return {self.arm_to_detector[i]: ucb[i] for i in range(self.n_arms)}

    def reset(self):
        """Reset AutoDD to initial state."""
        self.meta_extractor.reset()
        self.portfolio.reset_all()
        self.agent.reset()

        self._drift_detected = False
        self.current_detector = None
        self.current_arm = None
        self._n_detections = 0

        self.accuracy_buffer.clear()
        self.prev_window_accuracy = None

        self.selection_history = []
        self.drift_history = []
        self.reward_history = []

        self._n_instances = 0
        self._window_count = 0

        # Reset active detector state
        self._active_detector_name = None
        self._active_detector = None
        self._last_meta_features = None

    def save_state(self) -> Dict:
        """Save AutoDD state to dictionary."""
        return {
            'window_size': self.window_size,
            'alpha': self.alpha,
            'detector_names': self.portfolio.detector_names,
            'agent_state': self.agent.save_state(),
            'n_instances': self._n_instances,
            'n_detections': self._n_detections,
            'selection_history': self.selection_history,
            'drift_history': self.drift_history,
            'reward_history': self.reward_history
        }

    def load_state(self, state: Dict):
        """Load AutoDD state from dictionary."""
        self.agent.load_state(state['agent_state'])
        self._n_instances = state['n_instances']
        self._n_detections = state['n_detections']
        self.selection_history = state['selection_history']
        self.drift_history = state['drift_history']
        self.reward_history = state['reward_history']


class AutoDDClassifier:
    """
    AutoDD integrated with a base classifier.

    This class combines AutoDD drift detection with automatic model
    retraining upon drift detection.

    Parameters
    ----------
    base_classifier : river classifier
        The base classifier to use.
    window_size : int, default=100
        Window size for AutoDD.
    alpha : float, default=1.0
        LinUCB exploration parameter.

    Examples
    --------
    >>> from river import naive_bayes
    >>> from AutoDD import AutoDDClassifier
    >>> clf = AutoDDClassifier(base_classifier=naive_bayes.GaussianNB())
    >>> for x, y in stream:
    ...     y_pred = clf.predict_one(x)
    ...     clf.learn_one(x, y)
    """

    def __init__(
        self,
        base_classifier,
        window_size: int = 100,
        alpha: float = 1.0
    ):
        self.base_classifier = base_classifier
        self.classifier_class = type(base_classifier)

        self.autodd = AutoDD(window_size=window_size, alpha=alpha)

        self._n_drifts = 0

    def predict_one(self, x: Dict):
        """Predict label for one instance."""
        return self.base_classifier.predict_one(x)

    def learn_one(self, x: Dict, y) -> 'AutoDDClassifier':
        """
        Learn from one instance with automatic drift handling.

        Parameters
        ----------
        x : dict
            Feature dictionary.
        y : any
            True label.

        Returns
        -------
        self : AutoDDClassifier
        """
        # Get prediction
        y_pred = self.base_classifier.predict_one(x)

        # Update AutoDD
        self.autodd.update(x, y, y_pred)

        # Check for drift and reset if needed
        if self.autodd.drift_detected:
            self._n_drifts += 1
            self.base_classifier = self.classifier_class()

        # Learn from instance
        self.base_classifier.learn_one(x, y)

        return self

    @property
    def drift_detected(self) -> bool:
        """Whether drift was detected."""
        return self.autodd.drift_detected

    @property
    def n_drifts(self) -> int:
        """Number of drifts detected."""
        return self._n_drifts

    @property
    def current_detector(self) -> str:
        """Currently selected detector."""
        return self.autodd.current_detector

    def get_drift_info(self) -> Dict:
        """Get drift information."""
        return self.autodd.get_drift_info()
