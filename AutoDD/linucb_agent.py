"""
LinUCB Agent for AutoDD

Implements the LinUCB (Linear Upper Confidence Bound) contextual bandit
algorithm for adaptive drift detector selection.

LinUCB selects the detector with highest upper confidence bound:
    i_t = argmax_i (φ_t^T θ̂_i + α * sqrt(φ_t^T A_i^{-1} φ_t))

where:
    - φ_t: meta-feature vector (context)
    - θ̂_i: ridge regression estimate for arm i
    - A_i: design matrix for arm i
    - α: exploration parameter
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class LinUCBAgent:
    """
    LinUCB contextual bandit agent for detector selection.

    This agent learns the relationship between stream meta-features (context)
    and detector performance (reward), adaptively selecting the optimal detector.

    Parameters
    ----------
    n_arms : int
        Number of arms (detectors) in the bandit.
    n_features : int
        Dimension of the context (meta-features).
    alpha : float, default=1.0
        Exploration parameter controlling UCB width.
        Higher values encourage more exploration.
    lambda_reg : float, default=1.0
        Regularization parameter for ridge regression.

    Attributes
    ----------
    A : list of numpy.ndarray
        Design matrices for each arm (shape: n_features x n_features).
    b : list of numpy.ndarray
        Response vectors for each arm (shape: n_features,).
    theta : list of numpy.ndarray
        Estimated parameters for each arm (shape: n_features,).

    Examples
    --------
    >>> agent = LinUCBAgent(n_arms=7, n_features=15, alpha=1.0)
    >>> context = np.random.randn(15)
    >>> arm = agent.select_arm(context)
    >>> reward = 0.8  # observed reward
    >>> agent.update(arm, context, reward)

    References
    ----------
    Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th international conference on World wide web.
    """

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # Initialize design matrices A_i = λI
        self.A: List[np.ndarray] = [
            lambda_reg * np.eye(n_features) for _ in range(n_arms)
        ]

        # Initialize response vectors b_i = 0
        self.b: List[np.ndarray] = [
            np.zeros(n_features) for _ in range(n_arms)
        ]

        # Estimated parameters θ̂_i = A_i^{-1} b_i
        self.theta: List[np.ndarray] = [
            np.zeros(n_features) for _ in range(n_arms)
        ]

        # Inverse of design matrices (cached for efficiency)
        self.A_inv: List[np.ndarray] = [
            np.eye(n_features) / lambda_reg for _ in range(n_arms)
        ]

        # Statistics
        self.arm_counts = np.zeros(n_arms)
        self.total_reward = np.zeros(n_arms)
        self.n_rounds = 0

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm (detector) given the context (meta-features).

        Uses the UCB formula:
            UCB_i = φ^T θ̂_i + α * sqrt(φ^T A_i^{-1} φ)

        Parameters
        ----------
        context : numpy.ndarray
            Context vector (meta-features) of shape (n_features,).

        Returns
        -------
        arm : int
            Selected arm index (0 to n_arms-1).
        """
        context = np.asarray(context).flatten()

        if len(context) != self.n_features:
            raise ValueError(
                f"Context dimension {len(context)} != n_features {self.n_features}"
            )

        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            # Exploitation term: φ^T θ̂_i
            exploitation = np.dot(context, self.theta[i])

            # Exploration term: α * sqrt(φ^T A_i^{-1} φ)
            exploration = self.alpha * np.sqrt(
                np.dot(context, np.dot(self.A_inv[i], context))
            )

            ucb_values[i] = exploitation + exploration

        return int(np.argmax(ucb_values))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update the agent with observed reward.

        Updates the design matrix and response vector:
            A_i = A_i + φ φ^T
            b_i = b_i + r * φ
            θ̂_i = A_i^{-1} b_i

        Parameters
        ----------
        arm : int
            Selected arm index.
        context : numpy.ndarray
            Context vector.
        reward : float
            Observed reward.
        """
        context = np.asarray(context).flatten()

        # Update design matrix: A_i += φ φ^T
        self.A[arm] += np.outer(context, context)

        # Update response vector: b_i += r * φ
        self.b[arm] += reward * context

        # Update inverse using Sherman-Morrison formula for efficiency
        # (A + uv^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)
        A_inv = self.A_inv[arm]
        numerator = np.outer(np.dot(A_inv, context), np.dot(context, A_inv))
        denominator = 1 + np.dot(context, np.dot(A_inv, context))
        self.A_inv[arm] = A_inv - numerator / denominator

        # Update theta estimate: θ̂_i = A_i^{-1} b_i
        self.theta[arm] = np.dot(self.A_inv[arm], self.b[arm])

        # Update statistics
        self.arm_counts[arm] += 1
        self.total_reward[arm] += reward
        self.n_rounds += 1

    def get_ucb_values(self, context: np.ndarray) -> np.ndarray:
        """
        Get UCB values for all arms given context.

        Parameters
        ----------
        context : numpy.ndarray
            Context vector.

        Returns
        -------
        ucb_values : numpy.ndarray
            UCB values for each arm.
        """
        context = np.asarray(context).flatten()
        ucb_values = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            exploitation = np.dot(context, self.theta[i])
            exploration = self.alpha * np.sqrt(
                np.dot(context, np.dot(self.A_inv[i], context))
            )
            ucb_values[i] = exploitation + exploration

        return ucb_values

    def get_expected_rewards(self, context: np.ndarray) -> np.ndarray:
        """
        Get expected rewards (exploitation only) for all arms.

        Parameters
        ----------
        context : numpy.ndarray
            Context vector.

        Returns
        -------
        expected_rewards : numpy.ndarray
            Expected reward for each arm.
        """
        context = np.asarray(context).flatten()
        return np.array([np.dot(context, self.theta[i]) for i in range(self.n_arms)])

    def get_arm_stats(self) -> Dict:
        """
        Get statistics for all arms.

        Returns
        -------
        stats : dict
            Dictionary with arm statistics.
        """
        avg_rewards = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if self.arm_counts[i] > 0:
                avg_rewards[i] = self.total_reward[i] / self.arm_counts[i]

        return {
            'arm_counts': self.arm_counts.tolist(),
            'total_rewards': self.total_reward.tolist(),
            'average_rewards': avg_rewards.tolist(),
            'n_rounds': self.n_rounds
        }

    def get_regret_bound(self, T: int) -> float:
        """
        Calculate theoretical regret bound.

        From the paper, with probability at least 1-δ:
            R(T) ≤ 2α * sqrt(2mT * log(1 + TL²/m))

        where m is n_features, T is n_rounds.

        Parameters
        ----------
        T : int
            Number of rounds.

        Returns
        -------
        bound : float
            Regret bound.
        """
        m = self.n_features
        L = 1.0  # Assume bounded features ||φ|| ≤ L

        bound = 2 * self.alpha * np.sqrt(
            2 * m * T * np.log(1 + T * L**2 / m)
        )
        return bound

    def reset(self):
        """Reset the agent to initial state."""
        self.A = [
            self.lambda_reg * np.eye(self.n_features) for _ in range(self.n_arms)
        ]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.theta = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.A_inv = [
            np.eye(self.n_features) / self.lambda_reg for _ in range(self.n_arms)
        ]
        self.arm_counts = np.zeros(self.n_arms)
        self.total_reward = np.zeros(self.n_arms)
        self.n_rounds = 0

    def save_state(self) -> Dict:
        """Save agent state to dictionary."""
        return {
            'n_arms': self.n_arms,
            'n_features': self.n_features,
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'A': [a.tolist() for a in self.A],
            'b': [b.tolist() for b in self.b],
            'theta': [t.tolist() for t in self.theta],
            'arm_counts': self.arm_counts.tolist(),
            'total_reward': self.total_reward.tolist(),
            'n_rounds': self.n_rounds
        }

    def load_state(self, state: Dict):
        """Load agent state from dictionary."""
        self.n_arms = state['n_arms']
        self.n_features = state['n_features']
        self.alpha = state['alpha']
        self.lambda_reg = state['lambda_reg']
        self.A = [np.array(a) for a in state['A']]
        self.b = [np.array(b) for b in state['b']]
        self.theta = [np.array(t) for t in state['theta']]
        self.A_inv = [np.linalg.inv(a) for a in self.A]
        self.arm_counts = np.array(state['arm_counts'])
        self.total_reward = np.array(state['total_reward'])
        self.n_rounds = state['n_rounds']
