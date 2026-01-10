"""
Training utilities for Meta-ADD

Functions for training the prototypical network on drift streams.
"""

import numpy as np
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

from .meta_feature_extractor import MetaFeatureExtractor
from .prototypical_network import PrototypicalNetwork


def extract_meta_features_from_stream(
    error_sequence: List[float],
    window_size: int = 50,  # Match pretraining: n=50
    n_meta_features: int = 100  # Paper: l=100
) -> List[np.ndarray]:
    """
    Extract meta-features from an error sequence.

    Parameters
    ----------
    error_sequence : list of float
        Sequence of binary errors (0 or 1).
    window_size : int, default=50
        Window size for meta-feature extraction.
    n_meta_features : int, default=100
        Number of meta-features per sample.

    Returns
    -------
    features_list : list of numpy.ndarray
        List of extracted meta-feature arrays.
    """
    extractor = MetaFeatureExtractor(window_size)
    features_list = []

    for error in error_sequence:
        gap = extractor.update(error)

        if gap is not None:
            features = extractor.get_features(n_meta_features)
            if features is not None:
                features_list.append(features.copy())

    return features_list


def train_meta_detector(
    training_streams: List[Tuple[List[float], int]],
    window_size: int = 50,  # Match pretraining: n=50
    n_meta_features: int = 100,  # Paper: l=100
    embedding_dim: int = 128,
    hidden_dims: List[int] = None,  # Paper: 4-layer FCN
    n_support: int = 5,  # Paper: 5 support streams
    n_query: int = 15,  # Paper: 15 query streams
    n_epochs: int = 100,
    learning_rate: float = 1e-3,  # Paper: Adam lr=10^-3
    verbose: bool = True
) -> PrototypicalNetwork:
    """
    Train a meta-detector on multiple drift streams.

    Uses episodic training with support and query sets to learn
    prototype-based classification.

    Parameters
    ----------
    training_streams : list of (error_sequence, drift_type) tuples
        Training data where error_sequence is a list of binary errors
        and drift_type is an integer label (0=normal, 1=sudden, 2=gradual, 3=incremental).
    window_size : int, default=50
        Window size for meta-feature extraction.
    n_meta_features : int, default=100
        Number of meta-features per sample.
    embedding_dim : int, default=128
        Dimension of embedding space.
    hidden_dims : list of int, optional
        Hidden layer dimensions. Default is [256, 128].
    n_support : int, default=5
        Number of support examples per class per episode.
    n_query : int, default=15
        Number of query examples per class per episode.
    n_epochs : int, default=100
        Number of training epochs.
    learning_rate : float, default=1e-3
        Learning rate for optimizer.
    verbose : bool, default=True
        Whether to print training progress.

    Returns
    -------
    model : PrototypicalNetwork
        Trained prototypical network.

    Examples
    --------
    >>> # Prepare training data
    >>> training_data = [
    ...     (normal_errors, 0),
    ...     (sudden_errors, 1),
    ...     (gradual_errors, 2),
    ...     (incremental_errors, 3)
    ... ]
    >>> model = train_meta_detector(training_data, n_epochs=100)
    """
    if hidden_dims is None:
        hidden_dims = [256, 128]

    # Extract meta-features from all streams
    extracted_data = []

    if verbose:
        print("Extracting meta-features from training streams...")

    for error_sequence, drift_type in training_streams:
        features_list = extract_meta_features_from_stream(
            error_sequence, window_size, n_meta_features
        )
        for features in features_list:
            extracted_data.append((features, drift_type))

    if verbose:
        print(f"Extracted {len(extracted_data)} samples")

    # Group by class
    class_data = {i: [] for i in range(4)}
    for features, label in extracted_data:
        class_data[label].append(features)

    # Check minimum samples per class
    min_samples = min(len(data) for data in class_data.values())
    required_samples = n_support + n_query

    if min_samples < required_samples:
        raise ValueError(
            f"Not enough samples per class. Need at least {required_samples}, "
            f"but minimum is {min_samples}. Provide more training data."
        )

    if verbose:
        print(f"Samples per class: {[len(class_data[i]) for i in range(4)]}")

    # Initialize model
    model = PrototypicalNetwork(
        input_dim=n_meta_features,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        print(f"\nTraining meta-detector for {n_epochs} epochs...")

    best_loss = float('inf')

    for epoch in range(n_epochs):
        # Sample support and query sets (episodic training)
        support_set = []
        query_set = []

        for label, data in class_data.items():
            indices = np.random.choice(
                len(data), n_support + n_query, replace=False
            )
            support_indices = indices[:n_support]
            query_indices = indices[n_support:]

            for idx in support_indices:
                support_set.append((torch.FloatTensor(data[idx]), label))
            for idx in query_indices:
                query_set.append((torch.FloatTensor(data[idx]), label))

        # Compute prototypes from support set
        model.compute_prototypes(support_set)

        # Train on query set - accumulate losses then backward once
        optimizer.zero_grad()
        losses = []

        for features, label in query_set:
            embedding = model(features)

            # Calculate distances to prototypes
            distances = {}
            for l, prototype in model.prototypes.items():
                similarity = F.cosine_similarity(
                    embedding.unsqueeze(0),
                    prototype.unsqueeze(0),
                    dim=1
                )
                distances[l] = -similarity

            labels_list = list(distances.keys())
            dist_values = torch.stack([distances[l] for l in labels_list])

            log_probs = F.log_softmax(-dist_values, dim=0)
            target_idx = labels_list.index(label)
            loss = -log_probs[target_idx]
            losses.append(loss)

        # Sum all losses and backward once
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        optimizer.step()

        avg_loss = total_loss.item() / len(query_set)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    if verbose:
        print(f"\nTraining complete! Best loss: {best_loss:.4f}")

    # Compute final prototypes from all data
    all_support = [
        (torch.FloatTensor(features), label)
        for features, label in extracted_data
    ]
    model.compute_prototypes(all_support)

    return model


def generate_synthetic_training_data(
    n_streams_per_type: int = 20,
    n_samples_per_stream: int = 2000,
    seed: int = 42
) -> List[Tuple[List[float], int]]:
    """
    Generate synthetic training data for Meta-ADD.

    Creates error sequences that simulate different types of concept drift.

    Parameters
    ----------
    n_streams_per_type : int, default=20
        Number of streams to generate per drift type.
    n_samples_per_stream : int, default=2000
        Number of samples per stream.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    training_data : list of (error_sequence, drift_type) tuples
        Generated training data.
    """
    np.random.seed(seed)
    training_data = []

    # Generate normal streams (no drift)
    for _ in range(n_streams_per_type):
        base_error_rate = np.random.uniform(0.1, 0.3)
        errors = (np.random.random(n_samples_per_stream) < base_error_rate).astype(float)
        training_data.append((errors.tolist(), 0))

    # Generate sudden drift streams
    for _ in range(n_streams_per_type):
        drift_point = np.random.randint(n_samples_per_stream // 3, 2 * n_samples_per_stream // 3)
        error_rate_before = np.random.uniform(0.1, 0.25)
        error_rate_after = np.random.uniform(0.4, 0.6)

        errors = np.zeros(n_samples_per_stream)
        errors[:drift_point] = (np.random.random(drift_point) < error_rate_before).astype(float)
        errors[drift_point:] = (np.random.random(n_samples_per_stream - drift_point) < error_rate_after).astype(float)

        training_data.append((errors.tolist(), 1))

    # Generate gradual drift streams
    for _ in range(n_streams_per_type):
        drift_start = np.random.randint(n_samples_per_stream // 4, n_samples_per_stream // 2)
        drift_end = drift_start + np.random.randint(n_samples_per_stream // 4, n_samples_per_stream // 2)
        drift_end = min(drift_end, n_samples_per_stream)

        error_rate_before = np.random.uniform(0.1, 0.25)
        error_rate_after = np.random.uniform(0.4, 0.6)

        errors = np.zeros(n_samples_per_stream)

        for i in range(n_samples_per_stream):
            if i < drift_start:
                rate = error_rate_before
            elif i > drift_end:
                rate = error_rate_after
            else:
                # Linear interpolation
                progress = (i - drift_start) / (drift_end - drift_start)
                rate = error_rate_before + progress * (error_rate_after - error_rate_before)

            errors[i] = float(np.random.random() < rate)

        training_data.append((errors.tolist(), 2))

    # Generate incremental drift streams
    for _ in range(n_streams_per_type):
        error_rate_start = np.random.uniform(0.1, 0.2)
        error_rate_end = np.random.uniform(0.4, 0.6)

        errors = np.zeros(n_samples_per_stream)

        for i in range(n_samples_per_stream):
            progress = i / n_samples_per_stream
            rate = error_rate_start + progress * (error_rate_end - error_rate_start)
            # Add some noise
            rate += np.random.uniform(-0.02, 0.02)
            rate = np.clip(rate, 0, 1)
            errors[i] = float(np.random.random() < rate)

        training_data.append((errors.tolist(), 3))

    return training_data
