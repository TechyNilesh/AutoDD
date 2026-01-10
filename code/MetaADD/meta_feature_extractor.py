"""
Meta-Feature Extractor for Meta-ADD

Extracts meta-features (gaps between average error rates) from error sequences.
These meta-features capture the pattern of concept drift.
"""

import numpy as np
from collections import deque
from typing import Optional, List


class MetaFeatureExtractor:
    """
    Extracts meta-features from error rate sequences.

    Meta-features are the gaps between average error rates of consecutive
    sliding windows. These gaps capture the temporal pattern of errors
    which can indicate different types of concept drift.

    Parameters
    ----------
    window_size : int, default=50
        Number of samples in each window (n in the paper).

    Attributes
    ----------
    gaps : list
        List of computed gap values.
    previous_avg_error : float or None
        Average error rate from the previous window.

    Examples
    --------
    >>> extractor = MetaFeatureExtractor(window_size=50)
    >>> for error in errors:
    ...     gap = extractor.update(error)
    ...     if gap is not None:
    ...         print(f"Gap: {gap}")
    >>> features = extractor.get_features(n_features=100)
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.error_buffer = deque(maxlen=window_size)
        self.gaps: List[float] = []
        self.previous_avg_error: Optional[float] = None

    def update(self, error: float) -> Optional[float]:
        """
        Update with a new error value and return gap if window is full.

        Parameters
        ----------
        error : float
            Binary error value (0 for correct, 1 for incorrect prediction).

        Returns
        -------
        gap : float or None
            Gap between current and previous window average error rate,
            or None if window is not yet full.
        """
        self.error_buffer.append(error)

        if len(self.error_buffer) == self.window_size:
            # Calculate average error rate for current window
            current_avg_error = np.mean(list(self.error_buffer))

            gap = None
            if self.previous_avg_error is not None:
                # Calculate gap between consecutive windows
                gap = current_avg_error - self.previous_avg_error
                self.gaps.append(gap)

            self.previous_avg_error = current_avg_error

            # Clear buffer for next window
            self.error_buffer.clear()

            return gap

        return None

    def get_features(self, n_features: int = 100) -> Optional[np.ndarray]:
        """
        Get the last n_features gaps as meta-features.

        Parameters
        ----------
        n_features : int, default=100
            Number of meta-features to return (l in the paper).

        Returns
        -------
        features : numpy.ndarray or None
            Array of shape (n_features,) containing the meta-features,
            or None if not enough gaps have been collected.
        """
        if len(self.gaps) < n_features:
            return None

        return np.array(self.gaps[-n_features:], dtype=np.float32)

    def reset(self):
        """Reset the extractor to initial state."""
        self.error_buffer.clear()
        self.gaps = []
        self.previous_avg_error = None

    @property
    def n_gaps(self) -> int:
        """Return the number of gaps collected."""
        return len(self.gaps)

    def clone(self) -> 'MetaFeatureExtractor':
        """Create a clone of this extractor."""
        return MetaFeatureExtractor(window_size=self.window_size)
