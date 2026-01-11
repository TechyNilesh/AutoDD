"""
ADDM: Autoregressive-Based Drift Detection Method

Implementation of ADDM which uses Self-Exciting Threshold Autoregressive (SETAR)
models to detect concept drift by modeling the error rate as a time series.

The key insight is that the error rate can be modeled as a time series, and
regime changes in the SETAR model indicate concept drift.

Reference:
    Mayaki, M. Z. A., & Riveill, M. (2023).
    Autoregressive based drift detection method.
    arXiv:2203.04769v2

Paper: https://arxiv.org/abs/2203.04769
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import copy


class ADDM:
    """
    Autoregressive-Based Drift Detection Method (ADDM).

    Models the error rate as a time series using a simplified SETAR
    (Self-Exciting Threshold Autoregressive) model to detect regime
    changes indicating concept drift.

    The SETAR model partitions the error rate time series into regimes
    and detects drift when a regime change occurs.

    Parameters
    ----------
    window_size : int, default=100
        Size of the sliding window for error rate computation.
    lag : int, default=5
        Autoregressive lag order (p in the paper).
        The model uses error_rate[t-1], ..., error_rate[t-lag] to predict error_rate[t].
    threshold : float, default=2.0
        Number of standard deviations for drift detection.
        Higher = fewer false positives but may miss subtle drifts.
    min_instances : int, default=30
        Minimum instances before detection is enabled.
    update_interval : int, default=10
        How often to update error rate time series (every N instances).

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
    >>> from ADDM import ADDM
    >>> detector = ADDM(window_size=100, lag=5)
    >>> for i, (y_true, y_pred) in enumerate(stream):
    ...     error = 0 if y_true == y_pred else 1
    ...     detector.update(error)
    ...     if detector.drift_detected:
    ...         print(f"Drift detected at instance {i}")
    ...         severity = detector.get_drift_severity()
    ...         print(f"Drift severity: {severity:.3f}")

    References
    ----------
    .. [1] Mayaki, M. Z. A., & Riveill, M. (2023).
           Autoregressive based drift detection method.
           arXiv:2203.04769v2
    """

    def __init__(
        self,
        window_size: int = 100,
        lag: int = 5,
        threshold: float = 2.0,
        min_instances: int = 30,
        update_interval: int = 10
    ):
        self.window_size = window_size
        self.lag = lag
        self.threshold = threshold
        self.min_instances = min_instances
        self.update_interval = update_interval

        # Sliding window for errors (0 = correct, 1 = error)
        self._error_window: deque = deque(maxlen=window_size)

        # Time series of error rates (computed periodically)
        self._error_rates: deque = deque(maxlen=window_size)

        # State tracking
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

        # SETAR/AR model parameters (fitted using OLS)
        self._ar_coefficients: Optional[np.ndarray] = None
        self._residual_std: Optional[float] = None

        # Baseline statistics (first regime)
        self._baseline_mean: Optional[float] = None
        self._baseline_std: Optional[float] = None
        self._baseline_q3: Optional[float] = None  # Third quartile for severity

    def _compute_error_rate(self) -> float:
        """Compute current error rate from window."""
        if len(self._error_window) == 0:
            return 0.0
        return np.mean(list(self._error_window))

    def _fit_ar_model(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Fit AR(p) model using Ordinary Least Squares.

        From Equation 5 in the paper:
        β̂ = (X'X)^(-1) X'Y

        Returns
        -------
        coefficients : np.ndarray or None
            AR coefficients [φ_1, φ_2, ..., φ_p].
        residual_std : float or None
            Standard deviation of residuals (σ̂ from Eq. 6).
        """
        if len(self._error_rates) < self.lag + 10:
            return None, None

        rates = np.array(self._error_rates)
        n = len(rates)

        # Build design matrix X for AR(lag) model
        # X[t] = [1, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
        X = np.ones((n - self.lag, self.lag + 1))
        y = rates[self.lag:]

        for i in range(self.lag):
            X[:, i + 1] = rates[self.lag - 1 - i:n - 1 - i]

        # OLS estimation: β̂ = (X'X)^(-1) X'y
        try:
            XtX = X.T @ X
            # Add small regularization for numerical stability
            XtX_inv = np.linalg.inv(XtX + 1e-6 * np.eye(self.lag + 1))
            coefficients = XtX_inv @ X.T @ y

            # Compute residuals (Eq. 6)
            predictions = X @ coefficients
            residuals = y - predictions
            residual_std = np.std(residuals)

            return coefficients, residual_std

        except np.linalg.LinAlgError:
            return None, None

    def _predict_next(self) -> Optional[float]:
        """
        Predict next error rate using AR model.

        Returns predicted error rate or None if model not ready.
        """
        if self._ar_coefficients is None or len(self._error_rates) < self.lag:
            return None

        # Get most recent error rates [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
        recent_rates = list(self._error_rates)[-self.lag:][::-1]

        # Prediction: Ŷ_t = φ_0 + φ_1*Y_{t-1} + ... + φ_p*Y_{t-p}
        prediction = self._ar_coefficients[0]  # Intercept
        prediction += np.dot(self._ar_coefficients[1:], recent_rates)

        return prediction

    def _detect_regime_change(self, current_rate: float) -> Tuple[bool, bool]:
        """
        Detect regime change using SETAR-like threshold.

        The SETAR model assumes different AR models apply to different
        "regimes" of the time series. A significant prediction error
        indicates a regime change (drift).

        Returns
        -------
        drift : bool
            True if drift detected.
        warning : bool
            True if warning level reached.
        """
        if self._ar_coefficients is None or self._residual_std is None:
            return False, False

        predicted = self._predict_next()
        if predicted is None:
            return False, False

        # Compute prediction error
        prediction_error = abs(current_rate - predicted)

        # Drift if error exceeds threshold * residual_std
        drift = prediction_error > self.threshold * self._residual_std

        # Warning at 70% of threshold
        warning = prediction_error > 0.7 * self.threshold * self._residual_std

        return drift, warning and not drift

    def update(self, x: float) -> 'ADDM':
        """
        Update the detector with a new value.

        Parameters
        ----------
        x : float
            Prediction error indicator (0 = correct, 1 = error).

        Returns
        -------
        self : ADDM
            The detector instance.
        """
        self._n_updates += 1
        self._drift_detected = False
        self._warning_detected = False

        # Add to error window
        self._error_window.append(x)

        # Update error rate time series periodically
        if self._n_updates % self.update_interval == 0:
            current_rate = self._compute_error_rate()
            self._error_rates.append(current_rate)

            # Fit/update AR model periodically
            if len(self._error_rates) >= self.lag + 10:
                if self._n_updates % (self.update_interval * 5) == 0 or self._ar_coefficients is None:
                    self._ar_coefficients, self._residual_std = self._fit_ar_model()

                    # Set baseline if not set (first regime)
                    if self._baseline_mean is None and len(self._error_rates) >= 20:
                        baseline_rates = np.array(list(self._error_rates)[:20])
                        self._baseline_mean = np.mean(baseline_rates)
                        self._baseline_std = np.std(baseline_rates)
                        self._baseline_q3 = np.percentile(baseline_rates, 75)

                # Check for regime change
                if self._n_updates >= self.min_instances:
                    drift, warning = self._detect_regime_change(current_rate)

                    if drift:
                        self._drift_detected = True
                        self._n_detections += 1
                        self._reset_after_drift()
                    elif warning:
                        self._warning_detected = True

        return self

    def _reset_after_drift(self):
        """Reset internal state after drift detection."""
        self._error_window.clear()
        self._error_rates.clear()
        self._ar_coefficients = None
        self._residual_std = None
        self._baseline_mean = None
        self._baseline_std = None
        self._baseline_q3 = None

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

    def clone(self) -> 'ADDM':
        """Create a copy of this detector."""
        return copy.deepcopy(self)

    def reset(self):
        """Reset the detector to initial state."""
        self._error_window.clear()
        self._error_rates.clear()
        self._ar_coefficients = None
        self._residual_std = None
        self._baseline_mean = None
        self._baseline_std = None
        self._baseline_q3 = None
        self._drift_detected = False
        self._warning_detected = False
        self._n_detections = 0
        self._n_updates = 0

    def get_drift_severity(self) -> Optional[float]:
        """
        Estimate the severity of the detected drift.

        From Equation 8 in the paper:
        w_t = max(Q3^0, Q3^t) / (Q3^0 + Q3^t)

        where Q3 is the third quartile of error rates.

        Returns
        -------
        severity : float or None
            Drift severity in [0.5, 1.0], or None if not available.
            0.5 = no change, 1.0 = maximum severity.
        """
        if self._baseline_q3 is None or len(self._error_rates) < 10:
            return None

        # Current regime Q3
        current_rates = np.array(list(self._error_rates)[-10:])
        current_q3 = np.percentile(current_rates, 75)

        # Severity (Equation 8)
        q3_old = self._baseline_q3
        q3_new = current_q3

        if q3_old + q3_new == 0:
            return 0.5

        severity = max(q3_old, q3_new) / (q3_old + q3_new)
        return severity

    def get_current_error_rate(self) -> Optional[float]:
        """Get the most recent error rate."""
        if len(self._error_rates) == 0:
            return None
        return list(self._error_rates)[-1]

    def get_info(self) -> Dict[str, Any]:
        """Get detector information and current state."""
        return {
            'name': 'ADDM',
            'window_size': self.window_size,
            'lag': self.lag,
            'threshold': self.threshold,
            'n_updates': self._n_updates,
            'n_detections': self._n_detections,
            'current_error_rate': self.get_current_error_rate(),
            'drift_severity': self.get_drift_severity(),
            'ar_model_fitted': self._ar_coefficients is not None
        }

    def __repr__(self) -> str:
        return f"ADDM(window_size={self.window_size}, lag={self.lag}, threshold={self.threshold})"
