"""
Meta-ADD: Active Drift Detection with Meta Learning

Main drift detector class following River's base drift detector interface.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

import torch

from .meta_feature_extractor import MetaFeatureExtractor
from .prototypical_network import PrototypicalNetwork


class MetaADD:
    """
    Meta-ADD: Active Drift Detection with Meta Learning.

    This detector learns to classify different types of concept drift using
    a prototypical neural network trained via meta-learning. It can detect
    and distinguish between:
    - Normal (no drift)
    - Sudden drift
    - Gradual drift
    - Incremental drift

    The detector extracts meta-features (gaps between error rates) and uses
    a pre-trained or adaptable neural network to classify the drift type.

    Parameters
    ----------
    window_size : int, default=50
        Size of sliding window for error rate calculation.
    n_meta_features : int, default=100
        Number of meta-features (gaps) to use for classification.
    embedding_dim : int, default=128
        Dimension of the embedding space in the prototypical network.
    uncertainty_threshold : float, default=0.5
        Entropy threshold for active learning (warning zone).
    pretrained_model : PrototypicalNetwork, optional
        Pre-trained prototypical network. If None, a new network is created.

    Attributes
    ----------
    drift_detected : bool
        Whether a drift has been detected.
    warning_detected : bool
        Whether an uncertain prediction has been made (for active learning).
    current_drift_type : str
        The type of detected drift ('normal', 'sudden', 'gradual', 'incremental').
    confidence : float
        Confidence of the current prediction.
    n_detections : int
        Total number of drifts detected.

    Examples
    --------
    >>> from MetaADD import MetaADD
    >>> detector = MetaADD(window_size=50, n_meta_features=100)
    >>> for x, y_true in stream:
    ...     y_pred = model.predict_one(x)
    ...     detector.update(y_true, y_pred)
    ...     if detector.drift_detected:
    ...         print(f"Drift detected: {detector.current_drift_type}")

    References
    ----------
    Yu, H., Liu, T., Lu, J., & Zhang, G. (2021).
    Automatic Learning to Detect Concept Drift.
    arXiv preprint arXiv:2105.01419.
    """

    # Drift type mapping
    DRIFT_TYPES = {
        0: "normal",
        1: "sudden",
        2: "gradual",
        3: "incremental"
    }

    def __init__(
        self,
        window_size: int = 50,  # Match pretraining: n=50
        n_meta_features: int = 100,  # Paper: l=100
        embedding_dim: int = 128,
        uncertainty_threshold: float = 0.5,
        pretrained_model: Optional[PrototypicalNetwork] = None
    ):
        self.window_size = window_size
        self.n_meta_features = n_meta_features
        self.embedding_dim = embedding_dim
        self.uncertainty_threshold = uncertainty_threshold

        # Initialize meta-feature extractor
        self.feature_extractor = MetaFeatureExtractor(window_size)

        # Initialize or use pretrained model
        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = PrototypicalNetwork(
                input_dim=n_meta_features,
                embedding_dim=embedding_dim
            )
            # Initialize with default prototypes for cold start
            self._initialize_default_prototypes()

        # State tracking
        self._drift_detected = False
        self._warning_detected = False
        self.current_drift_type: Optional[str] = None
        self.confidence: Optional[float] = None
        self._n_detections = 0

        # Active learning buffers
        self.uncertain_samples: List[Tuple[np.ndarray, Optional[int]]] = []
        self.labeled_samples: List[Tuple[np.ndarray, int]] = []

        # Instance counter
        self._n_instances = 0

    def _initialize_default_prototypes(self):
        """Initialize default prototypes for cold start scenario."""
        # Create random prototypes for each drift type
        # These will be refined through active learning
        with torch.no_grad():
            for label in self.DRIFT_TYPES.keys():
                self.model.prototypes[label] = torch.randn(self.embedding_dim)

    def update(self, y_true, y_pred) -> 'MetaADD':
        """
        Update the detector with a new prediction result.

        Parameters
        ----------
        y_true : any
            True label.
        y_pred : any
            Predicted label.

        Returns
        -------
        self : MetaADD
            The detector instance.
        """
        self._n_instances += 1

        # Calculate error (binary)
        error = float(y_true != y_pred)

        # Update meta-feature extractor
        gap = self.feature_extractor.update(error)

        # Reset detection flags
        self._drift_detected = False
        self._warning_detected = False

        # Check if we have enough meta-features
        if gap is not None:
            meta_features = self.feature_extractor.get_features(self.n_meta_features)

            if meta_features is not None:
                self._classify_drift(meta_features)

        return self

    def _classify_drift(self, meta_features: np.ndarray):
        """Classify drift type using the prototypical network."""
        # Convert to tensor
        x_tensor = torch.FloatTensor(meta_features)

        # Get prediction
        with torch.no_grad():
            probs = self.model.predict_proba(x_tensor)
            predicted_class = self.model.predict(x_tensor)

        # Calculate entropy (uncertainty)
        entropy = -sum(p * np.log(p + 1e-10) for p in probs.values())

        # Store results
        self.current_drift_type = self.DRIFT_TYPES[predicted_class]
        self.confidence = max(probs.values())

        # Detect drift (anything other than normal)
        if predicted_class != 0:  # Not normal
            self._drift_detected = True
            self._n_detections += 1

            # Check uncertainty for active learning
            if entropy > self.uncertainty_threshold:
                self._warning_detected = True
                self.uncertain_samples.append((meta_features.copy(), None))

    @property
    def drift_detected(self) -> bool:
        """Whether a drift was detected in the last update."""
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        """Whether an uncertain prediction was made (for active learning)."""
        return self._warning_detected

    @property
    def n_detections(self) -> int:
        """Total number of drifts detected."""
        return self._n_detections

    def get_drift_info(self) -> Dict:
        """
        Get information about the current drift detection.

        Returns
        -------
        info : dict
            Dictionary containing 'drift_type', 'confidence', and 'n_detections'.
        """
        return {
            "drift_type": self.current_drift_type,
            "confidence": self.confidence,
            "n_detections": self._n_detections
        }

    def label_uncertain_sample(self, idx: int, label: int):
        """
        Label an uncertain sample for active learning.

        Parameters
        ----------
        idx : int
            Index of the uncertain sample to label.
        label : int
            True drift type label (0=normal, 1=sudden, 2=gradual, 3=incremental).
        """
        if idx < len(self.uncertain_samples):
            features, _ = self.uncertain_samples[idx]
            self.labeled_samples.append((features, label))
            self.uncertain_samples.pop(idx)

    def fine_tune(self, n_epochs: int = 10, learning_rate: float = 1e-3):
        """
        Fine-tune the model on labeled samples via active learning.

        Parameters
        ----------
        n_epochs : int, default=10
            Number of training epochs.
        learning_rate : float, default=1e-3
            Learning rate for optimizer.
        """
        if len(self.labeled_samples) == 0:
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            total_loss = 0

            # Update prototypes from labeled samples
            support_set = [
                (torch.FloatTensor(features), label)
                for features, label in self.labeled_samples
            ]
            self.model.compute_prototypes(support_set)

            # Train on labeled samples
            for features, label in self.labeled_samples:
                x = torch.FloatTensor(features)

                optimizer.zero_grad()

                # Get embedding
                embedding = self.model(x)

                # Calculate loss (negative log likelihood)
                distances = {}
                for l, prototype in self.model.prototypes.items():
                    similarity = torch.nn.functional.cosine_similarity(
                        embedding.unsqueeze(0),
                        prototype.unsqueeze(0),
                        dim=1
                    )
                    distances[l] = -similarity

                labels_list = list(distances.keys())
                dist_values = torch.stack([distances[l] for l in labels_list])

                log_probs = torch.nn.functional.log_softmax(-dist_values, dim=0)
                target_idx = labels_list.index(label)
                loss = -log_probs[target_idx]

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

    def reset(self):
        """Reset the detector to initial state."""
        self.feature_extractor.reset()
        self._drift_detected = False
        self._warning_detected = False
        self.current_drift_type = None
        self.confidence = None
        self._n_detections = 0
        self._n_instances = 0
        self.uncertain_samples = []
        # Keep labeled samples for future fine-tuning

    def clone(self) -> 'MetaADD':
        """Create a clone of this detector."""
        return MetaADD(
            window_size=self.window_size,
            n_meta_features=self.n_meta_features,
            embedding_dim=self.embedding_dim,
            uncertainty_threshold=self.uncertainty_threshold,
            pretrained_model=None  # Don't share model
        )

    def save_model(self, path: str):
        """Save the model to a file."""
        self.model.save(path)

    def load_model(self, path: str):
        """Load a pre-trained model from a file."""
        self.model = PrototypicalNetwork.load(path)
