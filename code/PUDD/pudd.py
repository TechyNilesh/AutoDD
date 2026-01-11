"""
PUDD: Prediction Uncertainty-based Drift Detector

Implementation of PUDD which uses the Prediction Uncertainty Index (PU-index)
to detect concept drift earlier than error rate-based methods.

The key insight is that prediction uncertainty changes before error rate does,
enabling earlier drift detection.

Reference:
    Lu, P., Lu, J., Liu, A., & Zhang, G. (2025).
    Early concept drift detection via prediction uncertainty.
    In AAAI Conference on Artificial Intelligence (AAAI-25).

Paper: https://arxiv.org/abs/2412.11158
Code: https://github.com/RocStone/PUDD
"""

import numpy as np
from typing import Optional, Dict, List, Union, Tuple, Any
from collections import deque
from scipy import stats
import copy


class PUDD:
    """
    Prediction Uncertainty-based Drift Detector (PUDD).

    Detects concept drift by monitoring changes in prediction uncertainty
    (PU-index) rather than error rate. The PU-index is defined as:

        u_i = 1 - f_{y_i}(x_i)

    where f_{y_i}(x_i) is the probability assigned to the true class.

    PUDD uses Chi-square test to compare PU-index histograms between
    a reference window and current window.

    Parameters
    ----------
    window_size : int, default=100
        Size of each window for comparison.
    n_bins : int, default=10
        Number of bins for histogram (K in the paper).
        The paper uses Ei-kMeans, but we use equal-width bins for simplicity.
    alpha : float, default=0.05
        Significance level for Chi-square test.
        Lower alpha = fewer false positives but may miss subtle drifts.
    min_instances : int, default=30
        Minimum instances before detection is enabled.

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
    >>> from PUDD import PUDD
    >>> detector = PUDD(window_size=100, alpha=0.05)
    >>> for x, y_true in stream:
    ...     y_proba = model.predict_proba(x)  # Get class probabilities
    ...     y_pred = model.predict(x)
    ...     # PU-index: 1 - probability of true class
    ...     pu_index = 1 - y_proba[y_true]
    ...     is_correct = (y_pred == y_true)
    ...     detector.update(pu_index, is_correct)
    ...     if detector.drift_detected:
    ...         print("Drift detected!")

    References
    ----------
    .. [1] Lu, P., Lu, J., Liu, A., & Zhang, G. (2025).
           Early concept drift detection via prediction uncertainty.
           In AAAI-25.
    """

    def __init__(
        self,
        window_size: int = 100,
        n_bins: int = 10,
        alpha: float = 0.05,
        min_instances: int = 30
    ):
        self.window_size = window_size
        self.n_bins = n_bins  # K in paper
        self.alpha = alpha
        self.min_instances = min_instances

        # Bin edges for PU-index histogram (0 to 1)
        # Paper uses Ei-kMeans for adaptive binning, we use equal-width
        self._bin_edges = np.linspace(0, 1, n_bins + 1)

        # Reference window (first window after reset)
        self._reference_pu: List[float] = []
        self._reference_correct: List[bool] = []
        self._reference_built = False

        # Current sliding window
        self._current_pu: deque = deque(maxlen=window_size)
        self._current_correct: deque = deque(maxlen=window_size)

        # State tracking
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

    def _build_contingency_table(
        self,
        pu_ref: List[float],
        correct_ref: List[bool],
        pu_cur: List[float],
        correct_cur: List[bool]
    ) -> np.ndarray:
        """
        Build contingency table for Chi-square test.

        From the paper (Section "PU-Index based Drift Detector"):
        - Correctly classified instances are binned into K bins
        - Misclassified instances go into a separate bin (K+1)

        Returns 2 x (K+1) contingency table.
        """
        K = self.n_bins

        # Separate correctly classified and misclassified
        pu_ref_correct = [pu for pu, c in zip(pu_ref, correct_ref) if c]
        pu_cur_correct = [pu for pu, c in zip(pu_cur, correct_cur) if c]
        n_ref_misclass = sum(1 for c in correct_ref if not c)
        n_cur_misclass = sum(1 for c in correct_cur if not c)

        # Build histogram for correctly classified instances
        hist_ref = np.histogram(pu_ref_correct, bins=self._bin_edges)[0]
        hist_cur = np.histogram(pu_cur_correct, bins=self._bin_edges)[0]

        # Contingency table: 2 rows (ref, cur) x (K+1) columns
        # Columns 0 to K-1: bins for correctly classified
        # Column K: misclassified count
        table = np.zeros((2, K + 1), dtype=float)
        table[0, :K] = hist_ref
        table[1, :K] = hist_cur
        table[0, K] = n_ref_misclass
        table[1, K] = n_cur_misclass

        return table

    def _chi_square_test(self, table: np.ndarray) -> float:
        """
        Perform Pearson's Chi-square test on contingency table.

        From Equations 12-14 in the paper.

        Returns
        -------
        p_value : float
            P-value of the test.
        """
        # Add small constant to avoid division by zero
        table = table + 0.5

        N = table.sum()
        if N == 0:
            return 1.0

        # Row and column sums
        row_sums = table.sum(axis=1, keepdims=True)
        col_sums = table.sum(axis=0, keepdims=True)

        # Expected frequencies (Equation 12)
        E = (row_sums * col_sums) / N

        # Chi-square statistic (Equation 13)
        chi2_stat = np.sum((table ** 2) / E) - N

        # Degrees of freedom
        df = self.n_bins  # (2-1) * (K+1-1) = K

        # P-value (Equation 14)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        return p_value

    def update(self, pu_index: float, is_correct: bool) -> 'PUDD':
        """
        Update the detector with a new instance.

        Parameters
        ----------
        pu_index : float
            Prediction Uncertainty Index: u_i = 1 - f_{y_i}(x_i)
            where f_{y_i}(x_i) is the probability of the true class.
            Value should be in [0, 1].
        is_correct : bool
            Whether the prediction was correct.

        Returns
        -------
        self : PUDD
            The detector instance.
        """
        self._n_updates += 1
        self._drift_detected = False
        self._warning_detected = False

        # Clip PU-index to valid range
        pu_index = np.clip(pu_index, 0.0, 1.0)

        # Build reference window first
        if not self._reference_built:
            self._reference_pu.append(pu_index)
            self._reference_correct.append(is_correct)

            if len(self._reference_pu) >= self.window_size:
                self._reference_built = True
            return self

        # Add to current window
        self._current_pu.append(pu_index)
        self._current_correct.append(is_correct)

        # Need full window for comparison
        if len(self._current_pu) < self.window_size:
            return self

        # Skip if not enough total instances
        if self._n_updates < self.min_instances:
            return self

        # Build contingency table
        table = self._build_contingency_table(
            self._reference_pu,
            self._reference_correct,
            list(self._current_pu),
            list(self._current_correct)
        )

        # Chi-square test
        p_value = self._chi_square_test(table)

        # Check for drift
        if p_value < self.alpha:
            self._drift_detected = True
            self._n_detections += 1
            self._reset_after_drift()
        elif p_value < self.alpha * 2:
            # Warning zone
            self._warning_detected = True

        return self

    def update_with_proba(
        self,
        y_true: int,
        y_pred: int,
        proba: Union[Dict, np.ndarray, List[float]]
    ) -> 'PUDD':
        """
        Convenience method to update with prediction probabilities.

        Computes PU-index automatically from class probabilities.

        Parameters
        ----------
        y_true : int
            True class label.
        y_pred : int
            Predicted class label.
        proba : dict, np.ndarray, or list
            Class probabilities. If dict, maps class labels to probabilities.
            If array/list, indexed by class label.

        Returns
        -------
        self : PUDD
        """
        # Get probability of true class
        if isinstance(proba, dict):
            prob_true = proba.get(y_true, 0.0)
        else:
            proba = np.array(proba)
            if y_true < len(proba):
                prob_true = proba[y_true]
            else:
                prob_true = 0.0

        # PU-index (Equation 7 in paper)
        pu_index = 1 - prob_true

        # Is prediction correct?
        is_correct = (y_pred == y_true)

        return self.update(pu_index, is_correct)

    def _reset_after_drift(self):
        """Reset windows after drift detection."""
        # Current window becomes new reference
        self._reference_pu = list(self._current_pu)
        self._reference_correct = list(self._current_correct)
        self._current_pu.clear()
        self._current_correct.clear()

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

    def clone(self) -> 'PUDD':
        """Create a copy of this detector."""
        return copy.deepcopy(self)

    def reset(self):
        """Reset the detector to initial state."""
        self._reference_pu = []
        self._reference_correct = []
        self._reference_built = False
        self._current_pu.clear()
        self._current_correct.clear()
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

    def get_current_uncertainty(self) -> Optional[float]:
        """Get mean PU-index of current window."""
        if len(self._current_pu) == 0:
            return None
        return np.mean(list(self._current_pu))

    def get_info(self) -> Dict[str, Any]:
        """Get detector information and current state."""
        return {
            'name': 'PUDD',
            'window_size': self.window_size,
            'n_bins': self.n_bins,
            'alpha': self.alpha,
            'n_updates': self._n_updates,
            'n_detections': self._n_detections,
            'current_uncertainty': self.get_current_uncertainty(),
            'reference_size': len(self._reference_pu),
            'current_size': len(self._current_pu)
        }

    def __repr__(self) -> str:
        return f"PUDD(window_size={self.window_size}, n_bins={self.n_bins}, alpha={self.alpha})"


class PUDDSimple:
    """
    Simplified PUDD that only requires error signal (for compatibility).

    This version uses prediction error (0/1) as a proxy and monitors
    error rate changes using the same Chi-square framework.

    Use the main PUDD class for better drift detection with full
    probability information.

    Parameters
    ----------
    window_size : int, default=100
        Size of the sliding window.
    alpha : float, default=0.05
        Significance level for Chi-square test.
    """

    def __init__(self, window_size: int = 100, alpha: float = 0.05):
        self.window_size = window_size
        self.alpha = alpha

        self._reference_errors: List[float] = []
        self._reference_built = False
        self._current_errors: deque = deque(maxlen=window_size)

        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

    def update(self, x: float) -> 'PUDDSimple':
        """
        Update with error signal (0 = correct, 1 = error).

        Parameters
        ----------
        x : float
            Error indicator (0 or 1).

        Returns
        -------
        self : PUDDSimple
        """
        self._n_updates += 1
        self._drift_detected = False
        self._warning_detected = False

        if not self._reference_built:
            self._reference_errors.append(x)
            if len(self._reference_errors) >= self.window_size:
                self._reference_built = True
            return self

        self._current_errors.append(x)

        if len(self._current_errors) < self.window_size:
            return self

        # Compare error rates using two-proportion z-test
        n1 = len(self._reference_errors)
        n2 = len(self._current_errors)
        p1 = np.mean(self._reference_errors)
        p2 = np.mean(list(self._current_errors))

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2) + 1e-10)

        # Z-statistic
        z = abs(p1 - p2) / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(z))

        if p_value < self.alpha:
            self._drift_detected = True
            self._n_detections += 1
            self._reference_errors = list(self._current_errors)
            self._current_errors.clear()
        elif p_value < self.alpha * 2:
            self._warning_detected = True

        return self

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        return self._warning_detected

    @property
    def n_detections(self) -> int:
        return self._n_detections

    def clone(self) -> 'PUDDSimple':
        return copy.deepcopy(self)

    def reset(self):
        self._reference_errors = []
        self._reference_built = False
        self._current_errors.clear()
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': 'PUDDSimple',
            'window_size': self.window_size,
            'alpha': self.alpha,
            'n_updates': self._n_updates,
            'n_detections': self._n_detections
        }

    def __repr__(self) -> str:
        return f"PUDDSimple(window_size={self.window_size}, alpha={self.alpha})"
