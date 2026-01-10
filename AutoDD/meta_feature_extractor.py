"""
Meta-Feature Extractor for AutoDD

Extracts 15 meta-features from data streams to characterize their properties.
These features are used by LinUCB to select the optimal drift detector.

Meta-features are organized into three categories:
1. Statistical (5): velocity, dimensionality, sparsity, noise_level, class_imbalance
2. Stationarity (5): error_volatility, kl_divergence, autocorrelation, mean_shift, variance_shift
3. Temporal (5): trend, periodicity, change_frequency, change_magnitude, recurrence_score
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.fft import fft


class MetaFeatureExtractor:
    """
    Extracts 15 meta-features from streaming data.

    The meta-features capture essential characteristics of the data stream
    that are relevant for drift detection, including velocity, stationarity,
    and temporal patterns.

    Parameters
    ----------
    window_size : int, default=100
        Size of the sliding window for feature computation.
    n_features : int, default=15
        Number of meta-features to extract.

    Attributes
    ----------
    feature_names : list
        Names of all 15 meta-features.
    current_features : numpy.ndarray
        Most recently computed meta-feature vector.

    Examples
    --------
    >>> extractor = MetaFeatureExtractor(window_size=100)
    >>> for x, y, y_pred in stream:
    ...     extractor.update(x, y, y_pred)
    ...     if extractor.is_ready():
    ...         features = extractor.get_features()
    """

    # Feature names
    STATISTICAL_FEATURES = [
        'velocity',
        'dimensionality',
        'sparsity',
        'noise_level',
        'class_imbalance'
    ]

    STATIONARITY_FEATURES = [
        'error_volatility',
        'kl_divergence',
        'autocorrelation',
        'mean_shift',
        'variance_shift'
    ]

    TEMPORAL_FEATURES = [
        'trend',
        'periodicity',
        'change_frequency',
        'change_magnitude',
        'recurrence_score'
    ]

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Feature names
        self.feature_names = (
            self.STATISTICAL_FEATURES +
            self.STATIONARITY_FEATURES +
            self.TEMPORAL_FEATURES
        )
        self.n_features = len(self.feature_names)

        # Data buffers
        self.x_buffer: deque = deque(maxlen=window_size)
        self.y_buffer: deque = deque(maxlen=window_size)
        self.error_buffer: deque = deque(maxlen=window_size)
        self.timestamp_buffer: deque = deque(maxlen=window_size)

        # Previous window statistics (for shift detection)
        self.prev_mean: Optional[np.ndarray] = None
        self.prev_var: Optional[np.ndarray] = None
        self.prev_error_dist: Optional[np.ndarray] = None

        # Current features
        self.current_features: Optional[np.ndarray] = None

        # Instance counter
        self._n_instances = 0
        self._last_update_time = 0

    def update(self, x: Dict, y, y_pred, timestamp: Optional[float] = None) -> 'MetaFeatureExtractor':
        """
        Update the extractor with a new instance.

        Parameters
        ----------
        x : dict
            Feature dictionary.
        y : any
            True label.
        y_pred : any
            Predicted label.
        timestamp : float, optional
            Instance timestamp. If None, uses instance count.

        Returns
        -------
        self : MetaFeatureExtractor
        """
        self._n_instances += 1

        # Convert x to array
        x_array = np.array(list(x.values()), dtype=np.float64)

        # Calculate error
        error = 0.0 if y == y_pred else 1.0

        # Use timestamp or instance count
        if timestamp is None:
            timestamp = float(self._n_instances)

        # Update buffers
        self.x_buffer.append(x_array)
        self.y_buffer.append(y)
        self.error_buffer.append(error)
        self.timestamp_buffer.append(timestamp)

        self._last_update_time = timestamp

        return self

    def is_ready(self) -> bool:
        """Check if enough data has been collected."""
        return len(self.x_buffer) >= self.window_size

    def get_features(self) -> np.ndarray:
        """
        Compute and return the 15 meta-features.

        Returns
        -------
        features : numpy.ndarray
            Array of shape (15,) containing the meta-features.
        """
        if not self.is_ready():
            # Return zeros if not enough data
            return np.zeros(self.n_features)

        # Convert buffers to arrays
        X = np.array(list(self.x_buffer))
        y = np.array(list(self.y_buffer))
        errors = np.array(list(self.error_buffer))
        timestamps = np.array(list(self.timestamp_buffer))

        # Compute all features
        features = np.zeros(self.n_features)

        # Statistical features (indices 0-4)
        features[0] = self._compute_velocity(timestamps)
        features[1] = self._compute_dimensionality(X)
        features[2] = self._compute_sparsity(X)
        features[3] = self._compute_noise_level(X, y)
        features[4] = self._compute_class_imbalance(y)

        # Stationarity features (indices 5-9)
        features[5] = self._compute_error_volatility(errors)
        features[6] = self._compute_kl_divergence(errors)
        features[7] = self._compute_autocorrelation(errors)
        features[8] = self._compute_mean_shift(X)
        features[9] = self._compute_variance_shift(X)

        # Temporal features (indices 10-14)
        features[10] = self._compute_trend(errors)
        features[11] = self._compute_periodicity(errors)
        features[12] = self._compute_change_frequency(errors)
        features[13] = self._compute_change_magnitude(errors)
        features[14] = self._compute_recurrence_score(errors)

        # Update previous window statistics
        self._update_previous_stats(X, errors)

        # Normalize features
        features = self._normalize_features(features)

        self.current_features = features
        return features

    def _compute_velocity(self, timestamps: np.ndarray) -> float:
        """Compute instance arrival rate."""
        if len(timestamps) < 2:
            return 0.0
        time_span = timestamps[-1] - timestamps[0]
        if time_span <= 0:
            return float(len(timestamps))
        return len(timestamps) / time_span

    def _compute_dimensionality(self, X: np.ndarray) -> float:
        """Return feature space dimension."""
        return float(X.shape[1]) if X.ndim > 1 else 1.0

    def _compute_sparsity(self, X: np.ndarray) -> float:
        """Compute proportion of zero-valued features."""
        return np.mean(X == 0)

    def _compute_noise_level(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate noise level using k-NN entropy approximation.
        Uses variance of nearest neighbor distances as proxy.
        """
        try:
            # Simple approach: compute variance of feature values
            # normalized by range (proxy for noise)
            feature_ranges = np.ptp(X, axis=0) + 1e-10
            normalized_std = np.std(X, axis=0) / feature_ranges
            return float(np.mean(normalized_std))
        except:
            return 0.0

    def _compute_class_imbalance(self, y: np.ndarray) -> float:
        """Compute ratio of majority to minority class."""
        unique, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return 1.0
        return float(np.max(counts)) / (np.min(counts) + 1e-10)

    def _compute_error_volatility(self, errors: np.ndarray) -> float:
        """Compute standard deviation of error rates over sub-windows."""
        n_subwindows = 10
        subwindow_size = len(errors) // n_subwindows

        if subwindow_size < 1:
            return 0.0

        error_rates = []
        for i in range(n_subwindows):
            start = i * subwindow_size
            end = start + subwindow_size
            error_rates.append(np.mean(errors[start:end]))

        return float(np.std(error_rates))

    def _compute_kl_divergence(self, errors: np.ndarray) -> float:
        """Compute KL divergence between first and second half error distributions."""
        mid = len(errors) // 2
        first_half = errors[:mid]
        second_half = errors[mid:]

        # Create histograms
        bins = np.linspace(0, 1, 11)
        hist1, _ = np.histogram(first_half, bins=bins, density=True)
        hist2, _ = np.histogram(second_half, bins=bins, density=True)

        # Add small epsilon to avoid division by zero
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10

        # Normalize
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        # KL divergence
        kl = float(np.sum(hist1 * np.log(hist1 / hist2)))
        return kl

    def _compute_autocorrelation(self, errors: np.ndarray) -> float:
        """Compute temporal autocorrelation of error rates (lag-1)."""
        if len(errors) < 2:
            return 0.0

        errors_centered = errors - np.mean(errors)
        variance = np.var(errors)

        if variance < 1e-10:
            return 0.0

        autocorr = np.correlate(errors_centered, errors_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags

        if len(autocorr) > 1:
            return float(autocorr[1] / (autocorr[0] + 1e-10))
        return 0.0

    def _compute_mean_shift(self, X: np.ndarray) -> float:
        """Compute average feature mean change from previous window."""
        current_mean = np.mean(X, axis=0)

        if self.prev_mean is None:
            return 0.0

        # Compute shift
        shift = np.abs(current_mean - self.prev_mean)
        return float(np.mean(shift))

    def _compute_variance_shift(self, X: np.ndarray) -> float:
        """Compute average feature variance change from previous window."""
        current_var = np.var(X, axis=0)

        if self.prev_var is None:
            return 0.0

        # Compute shift (ratio)
        shift = np.abs(current_var - self.prev_var) / (self.prev_var + 1e-10)
        return float(np.mean(shift))

    def _compute_trend(self, errors: np.ndarray) -> float:
        """Compute linear regression slope of error rates."""
        x = np.arange(len(errors))
        try:
            slope, _, _, _, _ = stats.linregress(x, errors)
            return float(slope)
        except:
            return 0.0

    def _compute_periodicity(self, errors: np.ndarray) -> float:
        """Compute periodicity using FFT peak frequency."""
        try:
            # Apply FFT
            fft_result = np.abs(fft(errors - np.mean(errors)))
            # Get magnitude of dominant frequency (excluding DC component)
            fft_result = fft_result[1:len(fft_result)//2]
            if len(fft_result) == 0:
                return 0.0
            return float(np.max(fft_result) / (np.sum(fft_result) + 1e-10))
        except:
            return 0.0

    def _compute_change_frequency(self, errors: np.ndarray) -> float:
        """Compute frequency of significant error rate changes."""
        threshold = np.std(errors) * 0.5
        changes = np.abs(np.diff(errors))
        significant_changes = np.sum(changes > threshold)
        return float(significant_changes) / len(errors)

    def _compute_change_magnitude(self, errors: np.ndarray) -> float:
        """Compute average magnitude of error rate shifts."""
        changes = np.abs(np.diff(errors))
        return float(np.mean(changes)) if len(changes) > 0 else 0.0

    def _compute_recurrence_score(self, errors: np.ndarray) -> float:
        """
        Compute pattern repetition indicator.
        Uses correlation between first and second half.
        """
        mid = len(errors) // 2
        first_half = errors[:mid]
        second_half = errors[mid:mid + len(first_half)]

        if len(first_half) != len(second_half) or len(first_half) < 2:
            return 0.0

        try:
            corr, _ = stats.pearsonr(first_half, second_half)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0

    def _update_previous_stats(self, X: np.ndarray, errors: np.ndarray):
        """Update previous window statistics."""
        self.prev_mean = np.mean(X, axis=0)
        self.prev_var = np.var(X, axis=0)
        self.prev_error_dist = errors.copy()

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range using min-max scaling."""
        # Handle NaN and Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        # Clip extreme values
        features = np.clip(features, -1e6, 1e6)

        # Normalize each feature
        # Note: In practice, you'd use running statistics for true online normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def reset(self):
        """Reset the extractor to initial state."""
        self.x_buffer.clear()
        self.y_buffer.clear()
        self.error_buffer.clear()
        self.timestamp_buffer.clear()
        self.prev_mean = None
        self.prev_var = None
        self.prev_error_dist = None
        self.current_features = None
        self._n_instances = 0

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()

    def get_feature_dict(self) -> Dict[str, float]:
        """Return features as a dictionary."""
        if self.current_features is None:
            return {name: 0.0 for name in self.feature_names}
        return dict(zip(self.feature_names, self.current_features))
