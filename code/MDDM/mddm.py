"""
MDDM: McDiarmid Drift Detection Method

Implementation of the McDiarmid Drift Detection Method (MDDM) for concept drift
detection in data streams. Uses McDiarmid's inequality with weighted sliding windows.

Reference:
    Pesaranghader, A., Viktor, H., & Paquet, E. (2018).
    McDiarmid drift detection methods for evolving data streams.
    In IJCNN 2018 (pp. 1-9). IEEE.

The paper proposes three weighting schemes:
- MDDM-A: Arithmetic weighting (linear)
- MDDM-G: Geometric weighting (exponential base 2)
- MDDM-E: Euler weighting (exponential base e)
"""

import numpy as np
from typing import Optional, Dict, Any
from collections import deque
import copy


class MDDM:
    """
    McDiarmid Drift Detection Method (MDDM).

    Uses McDiarmid's inequality to detect concept drift by comparing
    a weighted mean of recent predictions to the maximum weighted mean
    observed in the stream.

    The key insight is that weighted means give more importance to recent
    observations, making the detector more sensitive to recent changes.

    Parameters
    ----------
    n : int, default=100
        Size of the sliding window (called 'n' in the paper).
    delta : float, default=0.000001
        Confidence parameter for McDiarmid's bound (probability of false alarm).
        Smaller delta = higher confidence = fewer false positives.
    weighting : str, default='arithmetic'
        Weighting scheme: 'arithmetic', 'geometric', or 'euler'.
        - arithmetic: w_i = i (linear weights)
        - geometric: w_i = 2^i (exponential base 2)
        - euler: w_i = e^i (exponential base e)

    Attributes
    ----------
    drift_detected : bool
        True if drift was detected in the last update.
    warning_detected : bool
        True if warning level was reached.
    n_detections : int
        Total number of drifts detected.

    Examples
    --------
    >>> from MDDM import MDDM
    >>> detector = MDDM(n=100, delta=0.000001, weighting='arithmetic')
    >>> for i, (y_true, y_pred) in enumerate(stream):
    ...     error = 0 if y_true == y_pred else 1
    ...     detector.update(error)
    ...     if detector.drift_detected:
    ...         print(f"Drift detected at instance {i}")

    References
    ----------
    .. [1] Pesaranghader, A., Viktor, H., & Paquet, E. (2018).
           McDiarmid drift detection methods for evolving data streams.
           In 2018 International Joint Conference on Neural Networks (IJCNN)
           (pp. 1-9). IEEE.
    """

    def __init__(
        self,
        n: int = 100,
        delta: float = 0.000001,
        weighting: str = 'arithmetic'
    ):
        self.n = n
        self.delta = delta
        self.weighting = weighting

        # Validate weighting scheme
        if weighting not in ['arithmetic', 'geometric', 'euler']:
            raise ValueError(f"Unknown weighting scheme: {weighting}. "
                           f"Must be 'arithmetic', 'geometric', or 'euler'.")

        # Initialize weights based on scheme (Eq. 3, 4, 5 in paper)
        self._weights = self._compute_weights()
        self._weight_sum = np.sum(self._weights)

        # Compute McDiarmid's bound epsilon (Eq. 6 in paper)
        self._epsilon = self._compute_epsilon()

        # Sliding window for predictions (0 = correct, 1 = error)
        self._window: deque = deque(maxlen=n)

        # Maximum weighted mean of correct predictions observed
        self._max_weighted_mean = 0.0

        # State tracking
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

    def _compute_weights(self) -> np.ndarray:
        """
        Compute weights based on the weighting scheme.

        From the paper:
        - Arithmetic: w_i = i for i in [1, n]
        - Geometric: w_i = 2^(i-1) for i in [1, n]
        - Euler: w_i = e^(i-1) for i in [1, n]
        """
        n = self.n

        if self.weighting == 'arithmetic':
            # Eq. 3: w_i = i
            weights = np.arange(1, n + 1, dtype=np.float64)
        elif self.weighting == 'geometric':
            # Eq. 4: w_i = 2^(i-1)
            weights = np.power(2.0, np.arange(n, dtype=np.float64))
        else:  # euler
            # Eq. 5: w_i = e^(i-1)
            weights = np.exp(np.arange(n, dtype=np.float64))

        return weights

    def _compute_epsilon(self) -> float:
        """
        Compute McDiarmid's bound epsilon.

        From Eq. 6 in paper:
        epsilon = sqrt(-ln(delta) * sum(c_i^2) / 2)

        where c_i = w_i / sum(w) is the normalized weight,
        and the bound is on changes of c_i magnitude.
        """
        # Normalized weights
        c = self._weights / self._weight_sum

        # Sum of squared normalized weights
        sum_c_squared = np.sum(c ** 2)

        # McDiarmid's bound
        epsilon = np.sqrt(-np.log(self.delta) * sum_c_squared / 2)

        return epsilon

    def _compute_weighted_mean(self) -> float:
        """
        Compute weighted mean of correct predictions in current window.

        Returns the weighted mean mu_W = sum(w_i * X_i) / sum(w_i)
        where X_i = 1 if prediction was correct, 0 otherwise.
        """
        if len(self._window) < self.n:
            return 0.0

        # Window contains errors (1 = error), we want correct predictions
        window_array = np.array(self._window)
        correct = 1 - window_array  # Convert errors to correct predictions

        # Weighted mean
        weighted_sum = np.sum(correct * self._weights)
        return weighted_sum / self._weight_sum

    def update(self, x: float) -> 'MDDM':
        """
        Update the detector with a new value.

        Parameters
        ----------
        x : float
            Prediction error indicator (0 = correct, 1 = error).
            In River convention, this is typically the error signal.

        Returns
        -------
        self : MDDM
            The detector instance.
        """
        self._n_updates += 1
        self._drift_detected = False
        self._warning_detected = False

        # Add to sliding window
        self._window.append(x)

        # Need full window to compute meaningful statistics
        if len(self._window) < self.n:
            return self

        # Compute current weighted mean of correct predictions
        current_weighted_mean = self._compute_weighted_mean()

        # Update maximum weighted mean if current is higher (better performance)
        if current_weighted_mean > self._max_weighted_mean:
            self._max_weighted_mean = current_weighted_mean

        # Drift detection: check if performance dropped below threshold
        # From Eq. 7 in paper: detect if mu_max - mu_current >= epsilon
        difference = self._max_weighted_mean - current_weighted_mean

        if difference >= self._epsilon:
            self._drift_detected = True
            self._n_detections += 1
            self._reset_after_drift()
        elif difference >= self._epsilon * 0.7:
            # Warning zone (70% of drift threshold)
            self._warning_detected = True

        return self

    def _reset_after_drift(self):
        """Reset internal state after drift is detected."""
        self._window.clear()
        self._max_weighted_mean = 0.0

    @property
    def drift_detected(self) -> bool:
        """Whether drift was detected in the last update."""
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        """Whether warning was detected in the last update."""
        return self._warning_detected

    @property
    def n_detections(self) -> int:
        """Total number of drifts detected."""
        return self._n_detections

    def clone(self) -> 'MDDM':
        """Create a copy of this detector."""
        return copy.deepcopy(self)

    def reset(self):
        """Reset the detector to initial state."""
        self._window.clear()
        self._max_weighted_mean = 0.0
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

    def get_info(self) -> Dict[str, Any]:
        """Get detector information and current state."""
        return {
            'name': f'MDDM-{self.weighting[0].upper()}',
            'n': self.n,
            'delta': self.delta,
            'weighting': self.weighting,
            'epsilon': self._epsilon,
            'n_updates': self._n_updates,
            'n_detections': self._n_detections,
            'max_weighted_mean': self._max_weighted_mean,
            'window_size': len(self._window)
        }

    def __repr__(self) -> str:
        return f"MDDM(n={self.n}, delta={self.delta}, weighting='{self.weighting}')"


class MDDM_A(MDDM):
    """
    MDDM with Arithmetic (linear) weighting.

    Weights: w_i = i for i in [1, n]

    This variant gives linearly increasing importance to more recent observations.
    It's the least aggressive variant, suitable for gradual drifts.

    Parameters
    ----------
    n : int, default=100
        Size of the sliding window.
    delta : float, default=0.000001
        Confidence parameter for McDiarmid's bound.
    """

    def __init__(self, n: int = 100, delta: float = 0.000001):
        super().__init__(n=n, delta=delta, weighting='arithmetic')

    def __repr__(self) -> str:
        return f"MDDM_A(n={self.n}, delta={self.delta})"


class MDDM_G(MDDM):
    """
    MDDM with Geometric (base-2 exponential) weighting.

    Weights: w_i = 2^(i-1) for i in [1, n]

    This variant gives exponentially increasing importance to recent observations.
    It's more aggressive than arithmetic, good for moderate drift speeds.

    Parameters
    ----------
    n : int, default=100
        Size of the sliding window.
    delta : float, default=0.000001
        Confidence parameter for McDiarmid's bound.
    """

    def __init__(self, n: int = 100, delta: float = 0.000001):
        super().__init__(n=n, delta=delta, weighting='geometric')

    def __repr__(self) -> str:
        return f"MDDM_G(n={self.n}, delta={self.delta})"


class MDDM_E(MDDM):
    """
    MDDM with Euler (base-e exponential) weighting.

    Weights: w_i = e^(i-1) for i in [1, n]

    This variant gives the most aggressive exponential weighting.
    It's the most sensitive to recent changes, best for abrupt drifts.

    Parameters
    ----------
    n : int, default=100
        Size of the sliding window.
    delta : float, default=0.000001
        Confidence parameter for McDiarmid's bound.
    """

    def __init__(self, n: int = 100, delta: float = 0.000001):
        super().__init__(n=n, delta=delta, weighting='euler')

    def __repr__(self) -> str:
        return f"MDDM_E(n={self.n}, delta={self.delta})"
