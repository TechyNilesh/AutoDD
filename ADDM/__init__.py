"""
ADDM: Autoregressive-Based Drift Detection Method

A drift detection method that models the error rate as a time series
using Self-Exciting Threshold Autoregressive (SETAR) models.

Based on the paper: "Autoregressive Based Drift Detection Method"
by Mayaki & Riveill (IJCNN 2022)

Key Features:
- Treats error rate as a time series
- Uses SETAR model to detect regime changes
- Very low false alarm rate
- Theoretical guarantees through OLS estimation
- Can estimate drift severity

Usage:
    from ADDM import ADDM
    
    detector = ADDM(window_size=100, threshold=2.0)
    for x, y_true in stream:
        y_pred = model.predict_one(x)
        error = 0 if y_true == y_pred else 1
        detector.update(error)
        if detector.drift_detected:
            print("Drift detected!")
            model = reset_model()
        model.learn_one(x, y_true)
"""

from .addm import ADDM

__all__ = [
    'ADDM',
]

__version__ = '1.0.0'
