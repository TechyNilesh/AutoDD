"""
PUDD: Prediction Uncertainty-based Drift Detector

A drift detection method that uses prediction uncertainty (PU-index)
instead of error rate, enabling earlier drift detection.

Based on the paper: "Early Concept Drift Detection via Prediction Uncertainty"
by Lu et al. (AAAI 2025)

Key Features:
- Detects drift earlier than error rate-based methods
- Uses prediction uncertainty (entropy/confidence) as signal
- Can detect drift even when error rates remain stable
- Adaptive PU-index bucketing algorithm
- Uses Pearson's Chi-square test for detection

Usage:
    from PUDD import PUDD
    
    detector = PUDD(window_size=100, n_buckets=10)
    for x, y_true in stream:
        proba = model.predict_proba_one(x)  # Get prediction probabilities
        detector.update(proba)
        if detector.drift_detected:
            print("Drift detected!")
            model = reset_model()
        model.learn_one(x, y_true)
"""

from .pudd import PUDD, PUDDSimple

__all__ = [
    'PUDD',        # Full PUDD with PU-index and Chi-square test
    'PUDDSimple',  # Simplified version using only error signal
]

__version__ = '1.0.0'
