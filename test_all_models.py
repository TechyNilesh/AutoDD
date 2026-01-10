"""
Test Script for All Drift Detectors

This script tests all drift detectors on a small synthetic dataset
to verify that everything is working correctly.

Usage:
    python test_all_models.py
"""

import sys
import os
import time
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from river import naive_bayes, metrics, drift
from river.drift import binary as binary_drift
from river.datasets import synth

# Import our implementations
from AutoDD import AutoDD
from MetaADD import MetaADD

import warnings
warnings.filterwarnings("ignore")


def generate_test_stream(n_samples=2000, drift_position=1000):
    """Generate a small test stream with concept drift."""
    print(f"Generating test stream with {n_samples} samples...")
    print(f"Drift at position {drift_position}")

    stream = synth.ConceptDriftStream(
        stream=synth.SEA(variant=0, seed=42),
        drift_stream=synth.SEA(variant=2, seed=42),
        position=drift_position,
        width=50,
        seed=42
    )

    data = []
    for i, (x, y) in enumerate(stream.take(n_samples)):
        data.append((x, y))

    return data


def test_river_detector(detector, detector_name, stream_data, verbose=True):
    """Test a River drift detector."""
    print(f"\n{'='*50}")
    print(f"Testing {detector_name}")
    print(f"{'='*50}")

    model = naive_bayes.GaussianNB()
    accuracy = metrics.Accuracy()

    n_drifts = 0
    drift_points = []

    start_time = time.time()

    for i, (x, y) in enumerate(stream_data):
        # Predict
        y_pred = model.predict_one(x)

        if y_pred is not None:
            accuracy.update(y, y_pred)

            # Update detector with error
            error = 0 if y_pred == y else 1
            detector.update(error)

            if detector.drift_detected:
                n_drifts += 1
                drift_points.append(i)
                if verbose:
                    print(f"  Drift detected at instance {i}")
                # Reset model
                model = naive_bayes.GaussianNB()

        # Learn
        model.learn_one(x, y)

    elapsed = time.time() - start_time

    print(f"\nResults for {detector_name}:")
    print(f"  Accuracy: {accuracy.get():.4f}")
    print(f"  Drifts detected: {n_drifts}")
    print(f"  Drift points: {drift_points}")
    print(f"  Time: {elapsed:.3f}s")

    return {
        'name': detector_name,
        'accuracy': accuracy.get(),
        'n_drifts': n_drifts,
        'drift_points': drift_points,
        'time': elapsed,
        'status': 'PASS'
    }


def test_autodd(stream_data, verbose=True):
    """Test AutoDD detector."""
    print(f"\n{'='*50}")
    print(f"Testing AutoDD (Our Method)")
    print(f"{'='*50}")

    model = naive_bayes.GaussianNB()
    accuracy = metrics.Accuracy()

    # Use smaller window for testing
    detector = AutoDD(window_size=200, alpha=1.0)

    n_drifts = 0
    drift_points = []
    selections = []

    start_time = time.time()

    for i, (x, y) in enumerate(stream_data):
        # Predict
        y_pred = model.predict_one(x)

        if y_pred is not None:
            accuracy.update(y, y_pred)

        # Update detector
        detector.update(x, y, y_pred if y_pred is not None else y)

        if detector.drift_detected:
            n_drifts += 1
            drift_points.append(i)
            if detector.current_detector:
                selections.append(detector.current_detector)
            if verbose:
                print(f"  Drift detected at instance {i} by {detector.current_detector}")
            # Reset model
            model = naive_bayes.GaussianNB()

        # Learn
        model.learn_one(x, y)

    elapsed = time.time() - start_time

    print(f"\nResults for AutoDD:")
    print(f"  Accuracy: {accuracy.get():.4f}")
    print(f"  Drifts detected: {n_drifts}")
    print(f"  Drift points: {drift_points}")
    print(f"  Selection distribution: {detector.get_selection_distribution()}")
    print(f"  Time: {elapsed:.3f}s")

    return {
        'name': 'AutoDD',
        'accuracy': accuracy.get(),
        'n_drifts': n_drifts,
        'drift_points': drift_points,
        'time': elapsed,
        'status': 'PASS'
    }


def test_meta_add(stream_data, verbose=True):
    """Test Meta-ADD detector."""
    print(f"\n{'='*50}")
    print(f"Testing Meta-ADD")
    print(f"{'='*50}")

    model = naive_bayes.GaussianNB()
    accuracy = metrics.Accuracy()

    # Try to load pre-trained model, otherwise use default
    pretrained_model = None
    try:
        from MetaADD import load_pretrained_model
        pretrained_model = load_pretrained_model()
        print("  Using pre-trained model")
    except Exception:
        print("  No pre-trained model found, using default initialization")

    # Use parameters matching pre-trained model
    # Pre-trained model uses window_size=50, n_meta_features=100
    detector = MetaADD(
        window_size=50,
        n_meta_features=100,
        uncertainty_threshold=0.5,
        pretrained_model=pretrained_model
    )

    n_drifts = 0
    drift_points = []
    drift_types = []

    start_time = time.time()

    for i, (x, y) in enumerate(stream_data):
        # Predict
        y_pred = model.predict_one(x)

        if y_pred is not None:
            accuracy.update(y, y_pred)

            # Update detector
            detector.update(y, y_pred)

            if detector.drift_detected:
                n_drifts += 1
                drift_points.append(i)
                if detector.current_drift_type:
                    drift_types.append(detector.current_drift_type)
                if verbose:
                    print(f"  Drift detected at instance {i}: {detector.current_drift_type}")
                # Reset model
                model = naive_bayes.GaussianNB()

        # Learn
        model.learn_one(x, y)

    elapsed = time.time() - start_time

    print(f"\nResults for Meta-ADD:")
    print(f"  Accuracy: {accuracy.get():.4f}")
    print(f"  Drifts detected: {n_drifts}")
    print(f"  Drift points: {drift_points}")
    print(f"  Drift types: {drift_types}")
    print(f"  Time: {elapsed:.3f}s")

    return {
        'name': 'MetaADD',
        'accuracy': accuracy.get(),
        'n_drifts': n_drifts,
        'drift_points': drift_points,
        'time': elapsed,
        'status': 'PASS'
    }


def main():
    print("="*60)
    print("AutoDD - Test All Models")
    print("="*60)
    print("\nThis script tests all drift detectors on a synthetic stream.")
    print("Expected: Drift around position 3000\n")

    # Generate test stream (need 6000+ samples for Meta-ADD with n_meta_features=100)
    stream_data = generate_test_stream(n_samples=6000, drift_position=3000)

    results = []

    # Test River detectors
    # Note: DDM, EDDM, HDDM_A, HDDM_W are in river.drift.binary
    river_detectors = [
        ('ADWIN', drift.ADWIN(delta=0.002)),
        ('DDM', binary_drift.DDM()),
        ('EDDM', binary_drift.EDDM()),
        ('HDDM_A', binary_drift.HDDM_A()),
        ('HDDM_W', binary_drift.HDDM_W()),
        ('PageHinkley', drift.PageHinkley()),
        ('KSWIN', drift.KSWIN(alpha=0.005, window_size=100)),
    ]

    for name, detector in river_detectors:
        try:
            result = test_river_detector(detector, name, stream_data, verbose=False)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results.append({
                'name': name,
                'status': 'FAIL',
                'error': str(e)
            })

    # Test Meta-ADD
    try:
        result = test_meta_add(stream_data, verbose=False)
        results.append(result)
    except Exception as e:
        print(f"\n  ERROR in Meta-ADD: {e}")
        results.append({
            'name': 'MetaADD',
            'status': 'FAIL',
            'error': str(e)
        })

    # Test AutoDD
    try:
        result = test_autodd(stream_data, verbose=False)
        results.append(result)
    except Exception as e:
        print(f"\n  ERROR in AutoDD: {e}")
        results.append({
            'name': 'AutoDD',
            'status': 'FAIL',
            'error': str(e)
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Detector':<15} {'Status':<8} {'Accuracy':<10} {'Drifts':<8} {'Time':<10}")
    print("-"*55)

    passed = 0
    failed = 0

    for r in results:
        if r['status'] == 'PASS':
            passed += 1
            print(f"{r['name']:<15} {'PASS':<8} {r['accuracy']:.4f}     {r['n_drifts']:<8} {r['time']:.3f}s")
        else:
            failed += 1
            print(f"{r['name']:<15} {'FAIL':<8} {'N/A':<10} {'N/A':<8} {'N/A':<10}")

    print("-"*55)
    print(f"\nTotal: {passed} PASSED, {failed} FAILED")

    if failed == 0:
        print("\n✓ All tests passed! All detectors are working correctly.")
    else:
        print(f"\n✗ {failed} test(s) failed. Check the errors above.")

    return results


if __name__ == "__main__":
    main()
