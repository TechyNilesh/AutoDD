"""
MDDM: McDiarmid Drift Detection Method

A drift detection method that leverages McDiarmid's inequality to detect
changes in data distributions using weighted sliding windows.

Based on the paper: "McDiarmid Drift Detection Methods for Evolving Data Streams"
by Pesaranghader et al. (IJCNN 2017)

Key Features:
- Uses McDiarmid's inequality for change detection bounds
- Weighted sliding window emphasizing recent entries
- Shorter detection delays than traditional methods
- Low false negative rates

Usage:
    from MDDM import MDDM
    
    detector = MDDM(window_size=100, delta=0.01)
    for x, y_true in stream:
        y_pred = model.predict_one(x)
        error = 0 if y_true == y_pred else 1
        detector.update(error)
        if detector.drift_detected:
            print("Drift detected!")
            model = reset_model()
        model.learn_one(x, y_true)
"""

from .mddm import MDDM, MDDM_A, MDDM_G, MDDM_E

__all__ = [
    'MDDM',
    'MDDM_A',  # Arithmetic weighting
    'MDDM_G',  # Geometric weighting
    'MDDM_E',  # Euler weighting
]

__version__ = '1.0.0'
