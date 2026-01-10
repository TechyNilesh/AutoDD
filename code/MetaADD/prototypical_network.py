"""
Prototypical Neural Network for Meta-ADD

Implements the embedding network and prototype-based classification
for drift type detection.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Neural Network for drift classification.

    Maps input meta-features to an embedding space and classifies based on
    the nearest prototype (mean embedding of each class).

    Parameters
    ----------
    input_dim : int
        Dimension of input meta-features.
    embedding_dim : int, default=128
        Dimension of the embedding space.
    hidden_dims : list of int, default=[256, 128]
        List of hidden layer dimensions.
    dropout : float, default=0.2
        Dropout probability.

    Attributes
    ----------
    prototypes : dict
        Dictionary mapping class labels to prototype embeddings.
    embedding_network : nn.Sequential
        The neural network for embedding.

    Examples
    --------
    >>> model = PrototypicalNetwork(input_dim=100, embedding_dim=128)
    >>> x = torch.randn(1, 100)
    >>> embedding = model(x)
    >>> print(embedding.shape)  # torch.Size([1, 128])
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Build fully connected network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.embedding_network = nn.Sequential(*layers)
        self.prototypes: Dict[int, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed input features into embedding space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim) or (input_dim,).

        Returns
        -------
        embedding : torch.Tensor
            Embedded features of shape (batch_size, embedding_dim).
        """
        return self.embedding_network(x)

    def compute_prototypes(self, support_set: List[Tuple[torch.Tensor, int]]):
        """
        Compute prototypes for each class from support set.

        Parameters
        ----------
        support_set : list of (features, label) tuples
            Support examples for computing prototypes.
        """
        self.prototypes = {}
        class_embeddings: Dict[int, List[torch.Tensor]] = {}

        # Group embeddings by class
        for features, label in support_set:
            if features.dim() == 1:
                features = features.unsqueeze(0)
            embedding = self.forward(features).squeeze(0)

            if label not in class_embeddings:
                class_embeddings[label] = []
            class_embeddings[label].append(embedding)

        # Compute mean embedding (prototype) for each class
        for label, embeddings in class_embeddings.items():
            self.prototypes[label] = torch.mean(torch.stack(embeddings), dim=0)

    def predict_proba(self, x: torch.Tensor) -> Dict[int, float]:
        """
        Predict class probabilities using nearest prototype.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (input_dim,) or (1, input_dim).

        Returns
        -------
        probs : dict
            Dictionary mapping class labels to probabilities.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        embedding = self.forward(x).squeeze(0)

        # Calculate distances to all prototypes
        distances = {}
        for label, prototype in self.prototypes.items():
            # Use cosine similarity (converted to distance)
            similarity = F.cosine_similarity(
                embedding.unsqueeze(0),
                prototype.unsqueeze(0),
                dim=1
            )
            distances[label] = -similarity.item()

        # Convert distances to probabilities using softmax
        labels = list(distances.keys())
        dist_values = torch.tensor([distances[label] for label in labels])
        probs = F.softmax(-dist_values, dim=0)

        return {label: prob.item() for label, prob in zip(labels, probs)}

    def predict(self, x: torch.Tensor) -> int:
        """
        Predict class label.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        label : int
            Predicted class label.
        """
        probs = self.predict_proba(x)
        return max(probs, key=probs.get)

    def save(self, path: str):
        """Save model state to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'prototypes': self.prototypes,
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim
        }, path)

    @classmethod
    def load(cls, path: str) -> 'PrototypicalNetwork':
        """Load model from file."""
        checkpoint = torch.load(path)
        model = cls(
            input_dim=checkpoint['input_dim'],
            embedding_dim=checkpoint['embedding_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.prototypes = checkpoint['prototypes']
        return model
