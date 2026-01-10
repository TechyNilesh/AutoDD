"""
Synthetic Stream Generator for AutoDD Experiments

This script generates 100 synthetic data streams with different drift patterns
for training and testing the AutoDD framework (Experiments 1 & 2).

Drift Types:
- Sudden: Abrupt concept changes
- Gradual: Slow transitions between concepts
- Incremental: Continuous small shifts
- Recurring: Patterns that repeat over time

Usage:
    python generate_synthetic_streams.py [--output_dir OUTPUT_DIR] [--n_streams N] [--n_instances N]

Requirements:
    pip install river
"""

import os
import csv
import random
import argparse
from river.datasets import synth


def generate_sudden_drift_stream(stream_id, n_instances):
    """Generate a stream with sudden (abrupt) concept drift."""
    drift_position = random.randint(4000, 6000)
    stream = synth.ConceptDriftStream(
        stream=synth.Agrawal(classification_function=0, seed=stream_id),
        drift_stream=synth.Agrawal(classification_function=2, seed=stream_id + 100),
        position=drift_position,
        width=50  # Small width for sudden-like drift
    )
    return stream, {'drift_position': drift_position, 'width': 50}


def generate_gradual_drift_stream(stream_id, n_instances):
    """Generate a stream with gradual concept drift."""
    drift_position = random.randint(3000, 7000)
    width = random.randint(500, 1500)
    stream = synth.ConceptDriftStream(
        stream=synth.SEA(variant=0, seed=stream_id),
        drift_stream=synth.SEA(variant=3, seed=stream_id + 100),
        position=drift_position,
        width=width
    )
    return stream, {'drift_position': drift_position, 'width': width}


def generate_incremental_drift_stream(stream_id, n_instances):
    """Generate a stream with incremental concept drift."""
    stream = synth.Hyperplane(
        seed=stream_id,
        n_features=10,
        n_drift_features=5,
        mag_change=0.01,  # Small incremental changes
        noise_percentage=0.1
    )
    return stream, {'n_features': 10, 'n_drift_features': 5, 'mag_change': 0.01}


def generate_recurring_drift_stream(stream_id, n_instances):
    """Generate a stream with recurring concept drift."""
    stream = synth.RandomRBFDrift(
        seed_model=stream_id,
        seed_sample=stream_id + 50,
        n_classes=2,
        n_features=10,
        n_centroids=20,
        change_speed=0.001,
        n_drift_centroids=10
    )
    return stream, {'n_centroids': 20, 'change_speed': 0.001, 'n_drift_centroids': 10}


def save_stream_to_csv(stream, n_instances, filepath):
    """Save a stream to CSV file."""
    data = []
    feature_names = None

    for i, (x, y) in enumerate(stream.take(n_instances)):
        if feature_names is None:
            feature_names = list(x.keys()) + ['target']
        row = list(x.values()) + [y]
        data.append(row)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        writer.writerows(data)

    return len(data), len(feature_names) - 1


def generate_all_streams(output_dir, n_streams_per_type=25, n_instances=10000, seed=42):
    """
    Generate all synthetic streams for AutoDD experiments.

    Args:
        output_dir: Directory to save the generated streams
        n_streams_per_type: Number of streams per drift type (default: 25)
        n_instances: Number of instances per stream (default: 10000)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with generation metadata
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    drift_types = {
        'sudden': generate_sudden_drift_stream,
        'gradual': generate_gradual_drift_stream,
        'incremental': generate_incremental_drift_stream,
        'recurring': generate_recurring_drift_stream
    }

    metadata = {
        'total_streams': 0,
        'n_instances': n_instances,
        'streams': []
    }

    stream_id = 0

    for drift_type, generator_func in drift_types.items():
        print(f"\nGenerating {n_streams_per_type} {drift_type} drift streams...")

        for i in range(n_streams_per_type):
            stream_id += 1
            filename = f"stream_{stream_id:03d}_{drift_type}.csv"
            filepath = os.path.join(output_dir, filename)

            try:
                stream, params = generator_func(stream_id, n_instances)
                n_rows, n_features = save_stream_to_csv(stream, n_instances, filepath)

                stream_meta = {
                    'id': stream_id,
                    'filename': filename,
                    'drift_type': drift_type,
                    'n_instances': n_rows,
                    'n_features': n_features,
                    'params': params
                }
                metadata['streams'].append(stream_meta)
                metadata['total_streams'] += 1

                if stream_id % 10 == 0:
                    print(f"  Generated {stream_id} streams...")

            except Exception as e:
                print(f"  Error generating stream {stream_id}: {e}")
                continue

    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stream_id', 'filename', 'drift_type', 'n_instances', 'n_features', 'split'])
        for s in metadata['streams']:
            split = 'train' if s['id'] <= 70 else 'test'
            writer.writerow([s['id'], s['filename'], s['drift_type'], s['n_instances'], s['n_features'], split])

    return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic streams for AutoDD experiments')
    parser.add_argument('--output_dir', type=str, default='datasets/synthetic_streams',
                        help='Output directory for generated streams')
    parser.add_argument('--n_streams', type=int, default=25,
                        help='Number of streams per drift type (default: 25)')
    parser.add_argument('--n_instances', type=int, default=10000,
                        help='Number of instances per stream (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Stream Generator for AutoDD")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Streams per drift type: {args.n_streams}")
    print(f"Instances per stream: {args.n_instances}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    metadata = generate_all_streams(
        output_dir=args.output_dir,
        n_streams_per_type=args.n_streams,
        n_instances=args.n_instances,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total streams generated: {metadata['total_streams']}")
    print(f"  - Sudden drift: {args.n_streams}")
    print(f"  - Gradual drift: {args.n_streams}")
    print(f"  - Incremental drift: {args.n_streams}")
    print(f"  - Recurring drift: {args.n_streams}")
    print(f"\nTraining set: streams 001-070 (70 streams)")
    print(f"Test set: streams 071-100 (30 streams)")
    print(f"\nMetadata saved to: {os.path.join(args.output_dir, 'metadata.csv')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
